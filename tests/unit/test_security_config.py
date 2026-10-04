"""The API refuses unsafe configuration, and never reveals its secrets."""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.config import (
    PLACEHOLDER_SECRET_KEY,
    PLACEHOLDER_SUPABASE_KEY,
    PLACEHOLDER_SUPABASE_URL,
    Settings,
    configuration_errors,
    require_secure_settings,
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
GOOD_KEY = "unit-test-signing-key-" + "x" * 24
GOOD = {
    "secret_key": GOOD_KEY,
    "supabase_url": "https://example-project.supabase.co",
    "supabase_key": "unit-test-service-role-key",
    "cors_origins": "http://localhost:5173",
}


def settings(**overrides) -> Settings:
    # _env_file=None: these tests never read the developer's .env.
    return Settings(_env_file=None, **{**GOOD, **overrides})


@pytest.mark.parametrize("key", ["", "   ", PLACEHOLDER_SECRET_KEY, "replace-with-a-long-random-secret"])
def test_missing_or_placeholder_keys_stop_startup(key):
    with pytest.raises(RuntimeError, match="SECRET_KEY is not configured"):
        require_secure_settings(settings(secret_key=key))


def test_short_keys_stop_startup_without_revealing_the_key():
    with pytest.raises(RuntimeError, match="at least 32 characters") as raised:
        require_secure_settings(settings(secret_key="short-but-secret-value"))
    assert "short-but-secret-value" not in str(raised.value)


def test_a_configured_key_is_accepted():
    require_secure_settings(settings())


def test_only_hmac_signing_algorithms_are_accepted():
    assert configuration_errors(settings(algorithm="none"))
    assert not configuration_errors(settings(algorithm="HS512"))


@pytest.mark.parametrize("field,value", [
    ("supabase_url", PLACEHOLDER_SUPABASE_URL), ("supabase_url", ""), ("supabase_url", "not-a-url"),
    ("supabase_key", PLACEHOLDER_SUPABASE_KEY), ("supabase_key", " "),
])
def test_supabase_placeholders_stop_startup(field, value):
    with pytest.raises(RuntimeError, match=field.upper()):
        require_secure_settings(settings(**{field: value}))


@pytest.mark.parametrize("app_env", ["development", "production"])
def test_wildcard_cors_is_refused_everywhere(app_env):
    errors = configuration_errors(settings(app_env=app_env, cors_origins="*"))
    assert any("'*' is not allowed" in error for error in errors)


@pytest.mark.parametrize("origins", [
    "", "http://localhost:5173", "https://localhost:5173", "http://dashboard.example.org",
    "https://dashboard.example.org/app", "https://dashboard.example.org,http://127.0.0.1:5173",
])
def test_production_requires_exact_https_frontend_origins(origins):
    assert configuration_errors(settings(app_env="production", cors_origins=origins))


def test_production_accepts_exact_https_origins():
    assert configuration_errors(settings(
        app_env="production", cors_origins="https://dashboard.example.org, https://admin.example.org/",
    )) == []


def test_development_allows_the_local_frontend():
    assert configuration_errors(settings(cors_origins="http://localhost:5173,http://localhost:3000")) == []


def test_production_requires_https_supabase():
    errors = configuration_errors(settings(
        app_env="production", cors_origins="https://dashboard.example.org",
        supabase_url="http://db.example.org"))
    assert any("SUPABASE_URL must use https" in error for error in errors)


# ── The real application ─────────────────────────────────────────────────────

def test_the_app_does_not_start_with_the_placeholder_key(monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "settings", settings(secret_key=PLACEHOLDER_SECRET_KEY))
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        with TestClient(main.app):
            pass


def test_secrets_never_appear_in_logs_or_responses(caplog):
    """Start the app, call the public endpoints and a failed login: no configured secret leaks."""
    from app.core.config import get_settings
    from app.db.session import get_auth_supabase, get_supabase
    from app.main import app

    configured = get_settings()
    secrets = [configured.secret_key, configured.supabase_key]
    database = MagicMock()
    auth = MagicMock()
    auth.auth.sign_in_with_password.side_effect = Exception("Invalid login credentials")
    app.dependency_overrides[get_supabase] = lambda: database
    app.dependency_overrides[get_auth_supabase] = lambda: auth
    caplog.set_level(logging.DEBUG)
    try:
        with TestClient(app) as client:
            bodies = [
                client.get("/health").text,
                client.get("/health/models").text,
                client.get("/health/database-schema").text,
                client.post("/api/auth/login", json={"email": "x@example.org", "password": "wrong-password"}).text,
                client.get("/api/patients/my", headers={"Authorization": "Bearer not-a-token"}).text,
            ]
    finally:
        app.dependency_overrides.clear()
    for secret in secrets:
        assert secret and secret not in caplog.text
        assert all(secret not in body for body in bodies)
    assert "wrong-password" not in caplog.text


def _import_app(app_env: str) -> dict:
    """Import the app in a fresh interpreter with the given APP_ENV."""
    script = (
        "import json; from app.main import app;"
        "cors = next(m for m in app.user_middleware if m.cls.__name__ == 'CORSMiddleware');"
        "print(json.dumps({'docs': app.docs_url, 'redoc': app.redoc_url, 'openapi': app.openapi_url,"
        " 'credentials': cors.kwargs['allow_credentials'], 'origins': cors.kwargs['allow_origins'],"
        " 'methods': cors.kwargs['allow_methods']}))"
    )
    # SUPABASE_KEY stays the JWT-shaped test value from tests/conftest.py
    # (the Supabase client checks the key's shape when the app is imported).
    env = {**os.environ, "APP_ENV": app_env, "SECRET_KEY": GOOD_KEY,
           "CORS_ORIGINS": "https://dashboard.example.org", "SUPABASE_URL": "https://example-project.supabase.co"}
    result = subprocess.run([sys.executable, "-c", script], cwd=BACKEND_ROOT, env=env,
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_production_turns_off_the_interactive_docs():
    production = _import_app("production")
    assert production["docs"] is production["redoc"] is production["openapi"] is None
    development = _import_app("development")
    assert development["docs"] == "/docs" and development["openapi"] == "/openapi.json"


def test_cors_uses_exact_origins_without_credentials():
    cors = _import_app("production")
    assert cors["origins"] == ["https://dashboard.example.org"]
    assert cors["credentials"] is False
    assert "*" not in cors["methods"]
