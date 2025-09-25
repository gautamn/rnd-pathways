"""Configuration settings for the LangGraph demo application."""
import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")

    # Model Configuration
    model_name: str = Field("gpt-3.5-turbo", alias="MODEL_NAME")
    temperature: float = Field(0.7, alias="TEMPERATURE")
    max_tokens: int = Field(1000, alias="MAX_TOKENS")

    # Optional LLM Router Configuration
    use_llm_routing: bool = Field(False, alias="USE_LLM_ROUTING")
    router_model_name: Optional[str] = Field(None, alias="ROUTER_MODEL_NAME")
    router_temperature: float = Field(0.0, alias="ROUTER_TEMPERATURE")

    # Pydantic settings configuration (v2)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        populate_by_name=True,
        extra="ignore",
    )


def get_settings() -> Settings:
    """Get the application settings.

    Returns:
        Settings: The application settings
    """
    return Settings()
