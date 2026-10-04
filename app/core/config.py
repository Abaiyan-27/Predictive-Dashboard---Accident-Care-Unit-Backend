"""Application configuration loaded from the environment (and .env locally).

Every variable is documented in .env.example and DEPLOYMENT.md.
"""
from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

from pydantic_settings import BaseSettings, SettingsConfigDict


PLACEHOLDER_SECRET_KEY = "your-supabase-jwt-secret"
PLACEHOLDER_SUPABASE_URL = "https://your-project.supabase.co"
PLACEHOLDER_SUPABASE_KEY = "your-supabase-service-role-key"
# Placeholder values shown in .env.example and the documentation; never valid.
EXAMPLE_VALUES = {
    PLACEHOLDER_SECRET_KEY,
    PLACEHOLDER_SUPABASE_URL,
    PLACEHOLDER_SUPABASE_KEY,
    "replace-with-a-long-random-secret",
    "change-me",
}
MIN_SECRET_KEY_LENGTH = 32
SIGNING_ALGORITHMS = {"HS256", "HS384", "HS512"}
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # "development" (default) or "production". Production turns off the
    # interactive API docs and refuses local or wildcard CORS origins.
    app_env: Literal["development", "production"] = "development"
    log_level: str = "INFO"

    # Key the API signs its own session tokens with (HMAC). Server-only.
    # The default is a placeholder: the API refuses to start with it.
    secret_key: str = PLACEHOLDER_SECRET_KEY
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 480  # 8-hour hospital shift

    # Supabase. The key must be the service-role key (server-only): the
    # backend creates Auth accounts and reads tables protected by RLS.
    supabase_url: str = PLACEHOLDER_SUPABASE_URL
    supabase_key: str = PLACEHOLDER_SUPABASE_KEY

    # Comma-separated exact browser origins allowed to call the API.
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # Optional Hugging Face model repository. When configured, the backend
    # downloads this bundle at startup and runs the downloaded files locally.
    hf_model_repo: str | None = None
    hf_model_revision: str | None = None
    hf_token: str | None = None
    hf_model_cache_dir: str = "/tmp/accident-care-models"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def configuration_errors(settings: Settings) -> list[str]:
    """Every reason the API must not start with these settings (no values included)."""
    errors: list[str] = []

    key = (settings.secret_key or "").strip()
    if not key or key in EXAMPLE_VALUES:
        errors.append("SECRET_KEY is not configured (empty or a placeholder value).")
    elif len(key) < MIN_SECRET_KEY_LENGTH:
        errors.append(f"SECRET_KEY must be at least {MIN_SECRET_KEY_LENGTH} characters long.")
    if settings.algorithm not in SIGNING_ALGORITHMS:
        errors.append(f"ALGORITHM must be one of {', '.join(sorted(SIGNING_ALGORITHMS))}.")

    url = (settings.supabase_url or "").strip()
    if not url or url in EXAMPLE_VALUES or urlparse(url).scheme not in {"http", "https"}:
        errors.append("SUPABASE_URL is not configured (empty, a placeholder or not a URL).")
    supabase_key = (settings.supabase_key or "").strip()
    if not supabase_key or supabase_key in EXAMPLE_VALUES:
        errors.append("SUPABASE_KEY is not configured (empty or a placeholder value).")

    origins = settings.cors_origins_list
    if "*" in origins:
        errors.append("CORS_ORIGINS must list exact origins; '*' is not allowed.")
    if settings.is_production:
        if not origins:
            errors.append("CORS_ORIGINS must name the production frontend origin.")
        for origin in origins:
            parsed = urlparse(origin)
            if origin == "*":
                continue
            if parsed.scheme != "https" or not parsed.hostname or parsed.path or parsed.query:
                errors.append(f"CORS origin {origin!r} must be an exact https origin (no path).")
            elif parsed.hostname in LOOPBACK_HOSTS:
                errors.append(f"CORS origin {origin!r} is a local address; not allowed in production.")
        if urlparse(url).scheme == "http" and urlparse(url).hostname not in LOOPBACK_HOSTS:
            errors.append("SUPABASE_URL must use https in production.")
    return errors


def require_secure_settings(settings: Settings) -> None:
    """Refuse to run with missing, placeholder or unsafe configuration.

    The API signs its own session tokens with ``secret_key``; with the shipped
    placeholder (or a short key) anyone could forge a token for any role.
    The message names the variables only, never their values.
    """
    errors = configuration_errors(settings)
    if errors:
        raise RuntimeError(
            "The API configuration is not safe to start with:\n- " + "\n- ".join(errors)
            + "\nSee .env.example and DEPLOYMENT.md."
        )
