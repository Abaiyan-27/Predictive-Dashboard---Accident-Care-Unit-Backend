"""
FastAPI application entry point.
Registers CORS middleware, mounts all routers under /api prefix.
Connects to Supabase — no local database initialization needed.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from supabase import Client

from app.api import auth, nurse, severity, doctor, doctor_decisions, admin, government, xai
from app.core.config import get_settings, require_secure_settings
from app.db.session import get_supabase
from app.services import ml_service, schema_service

settings = get_settings()

# Application loggers (app.*) write to stderr; uvicorn keeps its own handlers.
# httpx logs every Supabase request URL at INFO, so it is kept at WARNING.
logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpcore", "hpack"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("app.startup")


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Check the configuration, then download the optional remote model bundle."""
    require_secure_settings(settings)
    logger.info("Environment: %s (API docs %s); %d CORS origin(s) allowed",
                settings.app_env, "off" if settings.is_production else "on", len(settings.cors_origins_list))
    _, message = ml_service.sync_models_from_huggingface(
        settings.hf_model_repo,
        token=settings.hf_token,
        cache_dir=settings.hf_model_cache_dir,
        revision=settings.hf_model_revision,
    )
    logger.info("Model startup: %s", message)
    not_ready = [name for name, state in ml_service.get_model_status().items() if not state["loadable"]]
    if not_ready:
        logger.warning("Models not ready (see /health/models): %s", ", ".join(not_ready))
    yield
    # shutdown logic goes here if needed


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Accident Care Unit — Predictive Dashboard API",
    description=(
        "REST backend for the Accident Care hospital dashboard. "
        "Powered by Supabase. Supports JWT auth, RBAC across 4 roles, "
        "ML-based patient triage, and audit logging."
    ),
    version="2.0.0",
    lifespan=lifespan,
    # The interactive docs describe every route; production does not serve them.
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    openapi_url=None if settings.is_production else "/openapi.json",
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    # Sessions travel in the Authorization header, never in cookies.
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Authorization", "Content-Type"],
    # Lists that are paged report their full size in this header.
    expose_headers=["X-Total-Count"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

API_PREFIX = "/api"

app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(nurse.router, prefix=API_PREFIX)
app.include_router(severity.router, prefix=API_PREFIX)
app.include_router(doctor.router, prefix=API_PREFIX)
app.include_router(doctor_decisions.router, prefix=API_PREFIX)
app.include_router(admin.router, prefix=API_PREFIX)
app.include_router(government.router, prefix=API_PREFIX)
app.include_router(xai.router, prefix=API_PREFIX)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "Accident Care API", "database": "Supabase"}


@app.get("/health/models", tags=["Health"])
def model_health_check():
    """Show which optional model artifacts are currently installed."""
    return ml_service.get_model_status()


@app.get("/health/database-schema", tags=["Health"])
def database_schema_health_check(supabase: Client = Depends(get_supabase)):
    """Report whether the live Supabase schema supports every integrated form."""
    return schema_service.get_model_integration_schema_status(supabase)
