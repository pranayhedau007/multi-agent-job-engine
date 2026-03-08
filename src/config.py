"""
Author: Pranay Hedau
Purpose: Centralized application configuration.
Uses pydantic-settings to load environment variables from .env file
into a typed, validated Python object. Every other module imports
settings from here instead of calling os.getenv() directly.
Usage:
    from src.config import settings
    print(settings.openai_api_key)
Date Created: 03-07-2026
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

"""Purpose: Application settings loaded from environment variables.

    Pydantic-settings automatically:
    - Reads from .env file (via model_config)
    - Converts types (e.g. QDRANT_PORT string "6333" → int 6333)
    - Raises ValidationError if required fields are missing
    - Matches env vars case-insensitively (OPENAI_API_KEY → openai_api_key)
    """
class Settings(BaseSettings):
    

    # --- LLM Provider ---
    llm_provider: str = "openai"  # "openai" or "ollama"

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # --- Tavily ---
    tavily_api_key: str = ""

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "job_matching"

    # --- App ---
    log_level: str = "INFO"

    # Tells pydantic-settings WHERE to find .env and how to read it
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """Cached settings loader — creates Settings object only once.

    @lru_cache means first call parses .env and creates Settings.
    Every subsequent call returns the same cached instance.
    This is Python's cleanest singleton pattern.
    """
    return Settings()


# Convenience alias — every file in the project will use:
#   from src.config import settings
settings = get_settings()