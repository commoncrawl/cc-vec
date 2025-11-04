#!/usr/bin/env python3
"""
Llama Stack Helper - Standalone script to manage Ollama + Llama Stack

This script can:
- Check and help install Ollama
- Pull required models
- Start Llama Stack (Docker or local via uv)
- Manage lifecycle (start/stop/status/logs)

Usage:
    uv run llama-stack-helper setup
    uv run llama-stack-helper start --backend docker
    uv run llama-stack-helper stop
    uv run llama-stack-helper status
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Model configuration (can be overridden via environment variables)
DEFAULT_INFERENCE_MODEL = "llama3.2:3b"
DEFAULT_EMBEDDING_MODEL = "toshk0/nomic-embed-text-v2-moe:Q6_K"
DEFAULT_EMBEDDING_DIMENSIONS = 768

# Get models from environment or use defaults
INFERENCE_MODEL = os.getenv("LLAMA_STACK_INFERENCE_MODEL", DEFAULT_INFERENCE_MODEL)
EMBEDDING_MODEL = os.getenv("LLAMA_STACK_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
EMBEDDING_DIMENSIONS = int(os.getenv("LLAMA_STACK_EMBEDDING_DIMENSIONS", str(DEFAULT_EMBEDDING_DIMENSIONS)))

# Required models based on configuration
REQUIRED_MODELS = {
    INFERENCE_MODEL: "Inference model",
    EMBEDDING_MODEL: f"Embedding model ({EMBEDDING_DIMENSIONS} dimensions)",
}

# Default configuration
DEFAULT_PORT = 8321
DEFAULT_DATA_DIR = Path.home() / ".llama"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Colors for output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'


def log_info(msg: str):
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")


def log_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")


def log_warn(msg: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")


def log_step(msg: str):
    print(f"{Colors.BLUE}[STEP]{Colors.NC} {msg}")


def log_success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.NC} {msg}")


class OllamaManager:
    """Manages Ollama installation and models"""

    def __init__(self, url: str = DEFAULT_OLLAMA_URL):
        self.url = url

    def is_running(self) -> bool:
        """Check if Ollama is accessible"""
        try:
            import requests
            response = requests.get(f"{self.url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available models"""
        try:
            import requests
            response = requests.get(f"{self.url}/api/tags", timeout=5)
            if response.status_code == 200:
                return [m["name"] for m in response.json().get("models", [])]
        except Exception:
            pass
        return []

    def has_model(self, model: str) -> bool:
        """Check if specific model is available"""
        models = self.list_models()
        return model in models

    def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama"""
        log_info(f"Pulling model: {model}")
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                check=True,
                capture_output=False
            )
            return result.returncode == 0
        except subprocess.CalledProcessError:
            log_error(f"Failed to pull model: {model}")
            return False
        except FileNotFoundError:
            log_error("Ollama CLI not found. Is Ollama installed?")
            return False

    def get_installation_instructions(self) -> str:
        """Get platform-specific installation instructions"""
        return """
Install Ollama:
  macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh
  Windows: Download from https://ollama.com/download

Then start Ollama:
  ollama serve

Or on macOS, Ollama runs automatically after installation.
"""


class DockerBackend:
    """Manages Llama Stack via Docker"""

    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self.container_name = "llama-stack-cc-vec"
        self.image = "llamastack/distribution-starter"
        self.config_path = self._get_config_path()

    def _get_config_path(self) -> Path:
        """Get config path"""
        script_dir = Path(__file__).parent
        config = script_dir / "conf" / "ollama-run.yaml"
        if not config.exists():
            log_error(f"Config not found: {config}")
            sys.exit(1)
        return config

    def check_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            subprocess.run(
                ["docker", "--version"],
                check=True,
                capture_output=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def is_running(self) -> bool:
        """Check if container is running"""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                check=True
            )
            return self.container_name in result.stdout
        except subprocess.CalledProcessError:
            return False

    def start(self) -> bool:
        """Start Docker container"""
        if self.is_running():
            log_warn("Llama Stack container already running")
            return True

        log_info(f"Starting Llama Stack (Docker) on port {self.port}")

        # Remove old container if exists
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True
        )

        try:
            subprocess.run([
                "docker", "run", "-d",
                "--name", self.container_name,
                "-p", f"{self.port}:{self.port}",
                "-v", f"{DEFAULT_DATA_DIR}:/root/.llama",
                "-v", f"{self.config_path}:/app/run.yaml",
                "-e", "RUN_CONFIG_PATH=/app/run.yaml",
                "-e", "LLAMA_STACK_DATA_DIR=/root/.llama",
                "-e", f"OLLAMA_URL=http://host.docker.internal:11434",
                "-e", f"LLAMA_STACK_INFERENCE_MODEL={INFERENCE_MODEL}",
                "-e", f"LLAMA_STACK_EMBEDDING_MODEL={EMBEDDING_MODEL}",
                "-e", f"LLAMA_STACK_EMBEDDING_DIMENSIONS={EMBEDDING_DIMENSIONS}",
                self.image,
                "--port", str(self.port)
            ], check=True)

            log_info("Waiting for Llama Stack to be ready...")
            if self._wait_for_health(timeout=30):
                log_success(f"Llama Stack running at http://localhost:{self.port}")
                return True
            else:
                log_error("Llama Stack failed to become healthy")
                return False

        except subprocess.CalledProcessError as e:
            log_error(f"Failed to start Docker container: {e}")
            return False

    def stop(self) -> bool:
        """Stop Docker container"""
        if not self.is_running():
            log_warn("Llama Stack container not running")
            return True

        try:
            subprocess.run(["docker", "stop", self.container_name], check=True)
            subprocess.run(["docker", "rm", self.container_name], check=True)
            log_success("Llama Stack stopped")
            return True
        except subprocess.CalledProcessError as e:
            log_error(f"Failed to stop container: {e}")
            return False

    def logs(self, tail: int = 20, follow: bool = False) -> None:
        """Show container logs"""
        cmd = ["docker", "logs"]
        if follow:
            cmd.append("-f")
        cmd.extend(["--tail", str(tail), self.container_name])

        try:
            subprocess.run(cmd)
        except subprocess.CalledProcessError:
            log_error("Failed to get logs")

    def _wait_for_health(self, timeout: int = 30) -> bool:
        """Wait for service to be healthy"""
        import requests
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(f"http://localhost:{self.port}/health", timeout=2)
                if response.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(2)
        return False


class LocalBackend:
    """Manages Llama Stack via local uv/venv"""

    def __init__(self, port: int = DEFAULT_PORT, data_dir: Path = DEFAULT_DATA_DIR):
        self.port = port
        self.data_dir = data_dir
        self.pid_file = self.data_dir / "llamastack.pid"
        self.log_file = self.data_dir / "llamastack.log"
        self.config_path = self._get_config_path()

    def _get_config_path(self) -> Path:
        """Get config path"""
        script_dir = Path(__file__).parent
        config = script_dir / "conf" / "ollama-run.yaml"
        if not config.exists():
            log_error(f"Config not found: {config}")
            sys.exit(1)
        return config

    def check_dependencies(self) -> bool:
        """Check if llama CLI is available"""
        try:
            subprocess.run(
                ["llama", "--version"],
                check=True,
                capture_output=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def setup(self) -> bool:
        """Install dependencies via uv"""
        log_step("Installing Llama Stack dependencies via uv...")

        try:
            subprocess.run([
                "uv", "pip", "install",
                "llama-stack",
                "llama-stack-client",
                "faiss-cpu",
                "aiosqlite",
                "sqlalchemy[asyncio]"
            ], check=True)

            # Create data directories
            self.data_dir.mkdir(parents=True, exist_ok=True)
            (self.data_dir / "files").mkdir(exist_ok=True)
            (self.data_dir / "providers.d").mkdir(exist_ok=True)

            log_success("Dependencies installed")
            return True

        except subprocess.CalledProcessError as e:
            log_error(f"Failed to install dependencies: {e}")
            return False

    def is_running(self) -> bool:
        """Check if process is running"""
        if not self.pid_file.exists():
            return False

        try:
            pid = int(self.pid_file.read_text())
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, ValueError):
            self.pid_file.unlink(missing_ok=True)
            return False

    def start(self) -> bool:
        """Start Llama Stack locally"""
        if self.is_running():
            log_warn("Llama Stack already running")
            return True

        if not self.check_dependencies():
            log_error("Llama Stack dependencies not installed")
            log_info("Run: uv run llama-stack-helper setup")
            return False

        log_info(f"Starting Llama Stack (local) on port {self.port}")

        # Set environment variables
        env = os.environ.copy()
        env["LLAMA_STACK_DATA_DIR"] = str(self.data_dir)
        env["LLAMA_STACK_PORT"] = str(self.port)
        env["OLLAMA_URL"] = DEFAULT_OLLAMA_URL
        env["LLAMA_STACK_INFERENCE_MODEL"] = INFERENCE_MODEL
        env["LLAMA_STACK_EMBEDDING_MODEL"] = EMBEDDING_MODEL
        env["LLAMA_STACK_EMBEDDING_DIMENSIONS"] = str(EMBEDDING_DIMENSIONS)

        try:
            # Start in background
            log_file = open(self.log_file, "w")
            process = subprocess.Popen(
                ["uv", "run", "llama", "stack", "run", str(self.config_path), "--port", str(self.port)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env
            )

            # Save PID
            self.pid_file.write_text(str(process.pid))

            log_info("Waiting for Llama Stack to be ready...")
            if self._wait_for_health(timeout=30):
                log_success(f"Llama Stack running at http://localhost:{self.port}")
                log_info(f"Logs: tail -f {self.log_file}")
                return True
            else:
                log_error("Llama Stack failed to become healthy")
                log_info(f"Check logs: tail -f {self.log_file}")
                return False

        except Exception as e:
            log_error(f"Failed to start Llama Stack: {e}")
            return False

    def stop(self) -> bool:
        """Stop Llama Stack"""
        if not self.is_running():
            log_warn("Llama Stack not running")
            return True

        try:
            pid = int(self.pid_file.read_text())
            os.kill(pid, signal.SIGTERM)

            # Wait for graceful shutdown
            for _ in range(10):
                try:
                    os.kill(pid, 0)
                    time.sleep(1)
                except ProcessLookupError:
                    break

            # Force kill if still running
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

            self.pid_file.unlink(missing_ok=True)
            log_success("Llama Stack stopped")
            return True

        except Exception as e:
            log_error(f"Failed to stop Llama Stack: {e}")
            return False

    def logs(self, tail: int = 20, follow: bool = False) -> None:
        """Show logs"""
        if not self.log_file.exists():
            log_warn("No log file found")
            return

        if follow:
            subprocess.run(["tail", "-f", str(self.log_file)])
        else:
            subprocess.run(["tail", "-n", str(tail), str(self.log_file)])

    def _wait_for_health(self, timeout: int = 30) -> bool:
        """Wait for service to be healthy"""
        import requests
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(f"http://localhost:{self.port}/health", timeout=2)
                if response.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(2)
        return False


def cmd_setup(args):
    """Setup: Check Ollama, pull models, install dependencies"""
    log_step("Step 1/3: Checking Ollama")

    ollama = OllamaManager(args.ollama_url)

    if not ollama.is_running():
        log_error("Ollama is not running")
        print(ollama.get_installation_instructions())
        sys.exit(1)

    log_success("Ollama is running")

    log_step("Step 2/3: Checking required models")

    for model, description in REQUIRED_MODELS.items():
        if ollama.has_model(model):
            log_success(f"{description}: {model}")
        else:
            log_warn(f"{description}: {model} not found")
            if args.non_interactive or input(f"Pull {model}? [Y/n]: ").lower() != 'n':
                if not ollama.pull_model(model):
                    log_error(f"Failed to pull {model}")
                    sys.exit(1)

    log_step("Step 3/3: Installing dependencies")

    if args.backend == "uv":
        local = LocalBackend(port=args.port)
        if not local.setup():
            sys.exit(1)
    else:
        docker = DockerBackend(port=args.port)
        if not docker.check_docker():
            log_error("Docker not found. Install Docker or use --backend uv")
            sys.exit(1)
        log_info("Docker backend ready (no additional setup needed)")

    print()
    log_success("Setup complete!")
    log_info(f"Start with: uv run llama-stack-helper start --backend {args.backend}")


def cmd_start(args):
    """Start Llama Stack"""
    # Check Ollama first
    ollama = OllamaManager(args.ollama_url)
    if not ollama.is_running():
        log_error("Ollama is not running. Start it first:")
        log_info("  ollama serve")
        sys.exit(1)

    # Check models
    missing = [m for m in REQUIRED_MODELS if not ollama.has_model(m)]
    if missing:
        log_error(f"Missing required models: {missing}")
        log_info("Run: uv run llama-stack-helper setup")
        sys.exit(1)

    # Start backend
    if args.backend == "docker":
        backend = DockerBackend(port=args.port)
        if not backend.check_docker():
            log_error("Docker not available. Use --backend uv")
            sys.exit(1)
        success = backend.start()
    else:
        backend = LocalBackend(port=args.port)
        success = backend.start()

    if success:
        print()
        log_success("Llama Stack is ready!")
        print()
        log_info("To use with cc-vec, run:")
        print(f'  eval "$(uv run llama-stack-helper env)"')
        print()
        log_info("Then use cc-vec normally with your Athena credentials")
    else:
        sys.exit(1)


def cmd_stop(args):
    """Stop Llama Stack"""
    if args.backend == "docker":
        backend = DockerBackend()
        if not backend.check_docker():
            log_error("Docker not available. Cannot stop Docker backend.")
            sys.exit(1)
        backend.stop()
    else:
        backend = LocalBackend()
        backend.stop()


def cmd_status(args):
    """Show status"""
    log_info("Checking status...")
    print()

    # Check Ollama
    ollama = OllamaManager(args.ollama_url)
    print("Ollama:")
    if ollama.is_running():
        log_success(f"  Running at {args.ollama_url}")
        models = ollama.list_models()
        print(f"  Models: {len(models)}")
        for model, desc in REQUIRED_MODELS.items():
            has = ollama.has_model(model)
            status = "✓" if has else "✗"
            print(f"    {status} {desc}: {model}")
    else:
        log_error(f"  Not running at {args.ollama_url}")

    print()

    # Check Llama Stack
    docker = DockerBackend(port=args.port)
    local = LocalBackend(port=args.port)

    print("Llama Stack:")

    # Check docker backend if docker is available
    docker_available = docker.check_docker()
    if docker_available and docker.is_running():
        log_success(f"  Running (Docker) at http://localhost:{args.port}")
    elif local.is_running():
        log_success(f"  Running (UV) at http://localhost:{args.port}")
    else:
        log_warn("  Not running")
        if not docker_available:
            print("    (Docker not available on this system)")


def cmd_logs(args):
    """Show logs"""
    docker = DockerBackend()
    local = LocalBackend()

    if docker.is_running():
        docker.logs(tail=args.tail, follow=args.follow)
    elif local.is_running():
        local.logs(tail=args.tail, follow=args.follow)
    else:
        log_error("Llama Stack not running")
        sys.exit(1)


def cmd_env(args):
    """Output environment variables for cc-vec"""
    docker = DockerBackend(port=args.port)
    local = LocalBackend(port=args.port)

    if not docker.is_running() and not local.is_running():
        log_error("Llama Stack is not running. Start it first with:")
        log_info("  uv run llama-stack-helper start")
        sys.exit(1)

    # Output in shell-sourceable format
    print(f"export OPENAI_BASE_URL=http://localhost:{args.port}/v1")
    print(f"export OPENAI_API_KEY=none")
    print(f"export OPENAI_VERIFY_SSL=false")
    print(f"export OPENAI_EMBEDDING_MODEL={EMBEDDING_MODEL}")
    print(f"export OPENAI_EMBEDDING_DIMENSIONS={EMBEDDING_DIMENSIONS}")




def main():
    parser = argparse.ArgumentParser(
        description="Manage Ollama + Llama Stack for cc-vec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First time setup (uses default models)
  uv run llama-stack-helper setup

  # Setup with custom models
  export LLAMA_STACK_INFERENCE_MODEL=llama3.2:1b
  export LLAMA_STACK_EMBEDDING_MODEL=nomic-embed-text
  export LLAMA_STACK_EMBEDDING_DIMENSIONS=768
  uv run llama-stack-helper setup

  # Start Llama Stack (Docker)
  uv run llama-stack-helper start --backend docker

  # Start Llama Stack (UV/Local)
  uv run llama-stack-helper start --backend uv

  # Check status
  uv run llama-stack-helper status

  # Set environment variables in your shell
  eval "$(uv run llama-stack-helper env)"

  # Now use cc-vec normally with Athena env vars + Llama Stack
  uv run cc-vec index --url-patterns "%.edu" --limit 10

  # View logs
  uv run llama-stack-helper logs --follow

  # Stop
  uv run llama-stack-helper stop

Environment Variables:
  LLAMA_STACK_INFERENCE_MODEL    Inference model (default: llama3.2:3b)
  LLAMA_STACK_EMBEDDING_MODEL    Embedding model (default: toshk0/nomic-embed-text-v2-moe:Q6_K)
  LLAMA_STACK_EMBEDDING_DIMENSIONS  Embedding dimensions (default: 768)
        """
    )

    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama URL")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Llama Stack port")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup Ollama and Llama Stack")
    setup_parser.add_argument("--backend", choices=["docker", "uv"], default="docker", help="Backend to use")
    setup_parser.add_argument("--non-interactive", action="store_true", help="Non-interactive mode")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start Llama Stack")
    start_parser.add_argument("--backend", choices=["docker", "uv"], default="docker", help="Backend to use")

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop Llama Stack")
    stop_parser.add_argument("--backend", choices=["docker", "uv"], default="docker", help="Backend to stop")

    # Status command
    subparsers.add_parser("status", help="Check status")

    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Show logs")
    logs_parser.add_argument("--tail", type=int, default=20, help="Number of lines to show")
    logs_parser.add_argument("--follow", "-f", action="store_true", help="Follow log output")

    # Env command
    subparsers.add_parser("env", help="Output environment variables for cc-vec")

    args = parser.parse_args()

    # Check for requests module
    try:
        import requests
    except ImportError:
        log_error("requests module not found. Install it with: uv pip install requests")
        sys.exit(1)

    # Dispatch commands
    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "start":
        cmd_start(args)
    elif args.command == "stop":
        cmd_stop(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "logs":
        cmd_logs(args)
    elif args.command == "env":
        cmd_env(args)


if __name__ == "__main__":
    main()
