"""
FastAPI application entry point.
Registers CORS middleware, mounts all routers under /api prefix, and
initialises the database with seed data on startup.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.db.init_db import init_db
from app.api import auth, nurse, doctor, admin, government

settings = get_settings()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run DB init on startup."""
    init_db()
    yield
    # shutdown logic goes here if needed


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Accident Care Unit — Predictive Dashboard API",
    description=(
        "Production-ready REST backend for the Accident Care hospital dashboard. "
        "Supports JWT auth, RBAC across 4 roles, ML-based patient triage, and audit logging."
    ),
    version="1.0.0",
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
    return {"status": "ok", "service": "Accident Care API"}
