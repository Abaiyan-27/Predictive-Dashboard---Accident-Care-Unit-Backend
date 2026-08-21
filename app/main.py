"""
FastAPI application entry point.
Registers CORS middleware, mounts all routers under /api prefix.
Connects to Supabase — no local database initialization needed.
"""
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from supabase import Client

from app.api import auth, nurse, doctor, admin, government
from app.core.config import get_settings
from app.db.session import get_supabase
from app.services import ml_service, schema_service

settings = get_settings()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle — Supabase manages the DB, so nothing to init."""
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
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

API_PREFIX = "/api"

app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(nurse.router, prefix=API_PREFIX)
app.include_router(doctor.router, prefix=API_PREFIX)
app.include_router(admin.router, prefix=API_PREFIX)
app.include_router(government.router, prefix=API_PREFIX)


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
