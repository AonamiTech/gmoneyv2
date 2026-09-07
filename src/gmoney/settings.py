from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GMONEY_", env_file=".env", extra="ignore")

    env: Literal["development", "test", "production"] = "development"
    tenant_id: str = "development"
    database_url: str = "postgresql+asyncpg://gmoney:gmoney@postgres:5432/gmoney"
    temporal_target: str = "temporal:7233"
    s3_endpoint: str = "http://minio:9000"
    s3_access_key: str = "gmoney"
    s3_secret_key: str = Field(default="change-me", repr=False)
    s3_bucket: str = "gmoney-v2"
    gemini_api_key: str = Field(
        default="",
        repr=False,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GMONEY_GEMINI_API_KEY"),
    )
    gemini_model: str = "gemini-3.5-flash"
    gemini_timeout_seconds: float = Field(default=120, gt=0, le=900)
    gemini_max_calls_per_document: int = Field(default=4, ge=0, le=100)
    gemini_max_cost_usd_per_document: float = Field(default=0.25, ge=0)
    gemini_input_cost_usd_per_million: float = Field(default=0, ge=0)
    gemini_output_cost_usd_per_million: float = Field(default=0, ge=0)
    gemini_prompt_version: str = "phase3-grounded-v1"
    gemini_redaction_version: str = "phase3-redaction-v1"
    gemini_promotion_path: Path | None = None
    table_selection_mode: Literal["off", "shadow", "enabled"] = "off"
    uvdoc_mode: Literal["off", "shadow", "enabled"] = "off"
    uvdoc_model_dir: Path | None = None
    uvdoc_preregistration_path: Path | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
