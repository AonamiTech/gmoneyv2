from functools import lru_cache
from typing import Literal

from pydantic import Field
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


@lru_cache
def get_settings() -> Settings:
    return Settings()

