#!/bin/bash
set -e

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            LLAMA_STACK_PORT="$2"
            shift 2
            ;;
        *)
            # Unknown option, pass through
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "=============================================="
echo "cc-vec-bot (llama-stack + ollama)"
echo "=============================================="
echo "LLAMA_STACK_PORT: ${LLAMA_STACK_PORT:-5001}"
echo "INFERENCE_MODEL:  ${INFERENCE_MODEL:-tinyllama}"
echo "OLLAMA_URL:       ${OLLAMA_URL:-<local>}"
echo "CHATBOT_PORT:     ${CHATBOT_PORT:-8008}"
echo "=============================================="

# Determine Ollama URL
if [ -z "$OLLAMA_URL" ]; then
    # No external Ollama URL provided - start local Ollama
    echo "Starting local Ollama server..."
    ollama serve &
    OLLAMA_PID=$!
    OLLAMA_URL="http://localhost:11434"
    export OLLAMA_URL

    # Wait for Ollama to be ready
    echo "Waiting for Ollama to start..."
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "Ollama is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "ERROR: Ollama failed to start"
            exit 1
        fi
        sleep 1
    done

    # Pull model if specified and PREFETCH_MODEL wasn't set at build time
    if [ -n "$INFERENCE_MODEL" ]; then
        echo "Checking if model needs to be pulled: $INFERENCE_MODEL"
        if ! ollama list | grep -q "^${INFERENCE_MODEL}"; then
            echo "Pulling model: $INFERENCE_MODEL"
            ollama pull "$INFERENCE_MODEL"
        else
            echo "Model $INFERENCE_MODEL already available"
        fi
    fi
else
    # External Ollama URL provided - verify connectivity
    echo "Using external Ollama at: $OLLAMA_URL"
    for i in {1..10}; do
        if curl -s "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
            echo "External Ollama is reachable"
            break
        fi
        if [ $i -eq 10 ]; then
            echo "WARNING: Cannot reach external Ollama at $OLLAMA_URL"
        fi
        sleep 1
    done
fi

# Start the chatbot in the background
echo "Starting chatbot-frontend on port ${CHATBOT_PORT}..."
(cd /opt/chatbot-frontend && uvicorn api:app --host 0.0.0.0 --port "${CHATBOT_PORT}" > /var/log/chatbot.log 2>&1 &)
CHATBOT_PID=$!
echo "Chatbot-frontend started with PID ${CHATBOT_PID}"

# Give chatbot a moment to start
sleep 2

# Start llama-stack server
# The distribution-starter base image includes llama-stack
echo "Starting llama-stack on port ${LLAMA_STACK_PORT}..."
export OLLAMA_URL="${OLLAMA_URL}"
export INFERENCE_MODEL="${INFERENCE_MODEL}"
exec llama stack run starter \
    --port "${LLAMA_STACK_PORT}" \
    "${EXTRA_ARGS[@]}"

