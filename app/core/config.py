"""Application configuration loaded from .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # JWT (Supabase JWT secret for validating tokens)
    secret_key: str = "your-supabase-jwt-secret"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 480  # 8-hour hospital shift

    # Supabase
    supabase_url: str = "https://your-project.supabase.co"
    supabase_key: str = "your-supabase-service-role-key"

    # CORS
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
