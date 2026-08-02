"""Application configuration loaded from .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # JWT
    secret_key: str = "accident-care-super-secret-jwt-key-change-in-production-2024"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 480  # 8-hour hospital shift

    # Database
    database_url: str = "sqlite:///./accident_care.db"

    # CORS
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
