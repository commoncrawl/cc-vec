"""OpenAI configuration for cc-vec."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OpenAISettings:
    """OpenAI API configuration."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_dimensions: Optional[int] = None
    verify_ssl: bool = True

    def is_configured(self) -> bool:
        """Check if OpenAI is properly configured."""
        return bool(self.api_key)
