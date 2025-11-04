"""Main configuration management for cc-vec."""

import logging
import os
from dataclasses import dataclass

from .athena_config import AthenaSettings
from .openai_config import OpenAISettings
from .logging_config import LoggingSettings


@dataclass
class CCVecConfig:
    """Main cc-vec configuration."""

    athena: AthenaSettings
    openai: OpenAISettings
    logging: LoggingSettings

    @classmethod
    def from_env(cls) -> "CCVecConfig":
        """Create configuration from environment variables."""
        return cls(
            athena=AthenaSettings(
                output_bucket=os.getenv("ATHENA_OUTPUT_BUCKET"),
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-west-2"),
                max_results=int(os.getenv("ATHENA_MAX_RESULTS", "100")),
                timeout_seconds=int(os.getenv("ATHENA_TIMEOUT", "120")),
            ),
            openai=OpenAISettings(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL"),
                embedding_dimensions=int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS"))
                if os.getenv("OPENAI_EMBEDDING_DIMENSIONS")
                else None,
                verify_ssl=os.getenv("OPENAI_VERIFY_SSL", "true").lower()
                not in ["false", "0", "no"],
            ),
            logging=LoggingSettings(
                level=os.getenv("LOG_LEVEL", "INFO"),
                format=os.getenv(
                    "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                ),
            ),
        )

    def setup_logging(self) -> None:
        """Configure logging based on settings."""
        logging.basicConfig(
            level=self.logging.log_level,
            format=self.logging.format,
            force=True,  # Override existing configuration
        )

        # Log configuration status
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured at {self.logging.level} level")

        if self.athena.is_configured():
            logger.info(f"Athena backend configured: {self.athena.output_bucket}")

        if self.openai.is_configured():
            logger.info("OpenAI client configured")
            if self.openai.base_url:
                logger.info(f"Using custom OpenAI base URL: {self.openai.base_url}")
            if not self.openai.verify_ssl:
                logger.warning(
                    "⚠️  SSL certificate verification is DISABLED. "
                    "This should only be used in development with trusted endpoints."
                )
        else:
            logger.warning(
                "OpenAI API key not configured - vector operations unavailable"
            )


def load_config() -> CCVecConfig:
    """Load cc-vec configuration from environment variables.

    Returns:
        CCVecConfig: Loaded configuration

    Raises:
        ValueError: If ATHENA_OUTPUT_BUCKET is not configured
    """
    config = CCVecConfig.from_env()

    # Fail early if Athena is not configured
    if not config.athena.is_configured():
        raise ValueError(
            "ATHENA_OUTPUT_BUCKET environment variable is required. "
            "Please set it to an S3 bucket path (e.g., 's3://your-bucket/athena-results/')"
        )

    return config
