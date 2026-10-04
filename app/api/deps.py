"""
Shared FastAPI dependencies
===========================
- get_current_user  — extracts + validates JWT, returns user dict
- get_principal     — the caller as an authorization Principal
- RoleChecker       — callable dependency that enforces allowed roles
                      (hospital and record scoping: app.core.authorization)
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, Query, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from supabase import Client

from app.core.authorization import (
    ADMIN,
    DOCTOR,
    GOVERNMENT,
    NURSE,
    Principal,
    ensure_assigned_hospital,
    ensure_role,
    principal_from_user,
)
from app.core.security import decode_token
from app.db.session import get_supabase

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    supabase: Client = Depends(get_supabase),
) -> dict:
    """Validate JWT and return the authenticated user as a dict."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_token(credentials.credentials)
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Try to find user in User table first
    result = supabase.table("User").select("*").eq("user_id", user_id).execute()
    if result.data:
        user = result.data[0]
        user["_type"] = "user"
        return user

    # Try Government_Officer table
    result = supabase.table("Government_Officer").select("*").eq("officer_id", user_id).execute()
    if result.data:
        officer = result.data[0]
        officer["_type"] = "government_officer"
        officer["role"] = "government"
        return officer

    raise credentials_exception


def get_principal(current_user: dict = Depends(get_current_user)) -> Principal:
    """The caller's identity, role and hospital as currently stored in the database."""
    return principal_from_user(current_user)


class RoleChecker:
    """Dependency that admits only the given roles.

    Staff roles also require an assigned hospital, because every staff
    endpoint reads or writes hospital-scoped data.
    """

    def __init__(self, allowed_roles: list[str]) -> None:
        self.allowed_roles = allowed_roles

    def __call__(self, principal: Principal = Depends(get_principal)) -> Principal:
        ensure_role(principal, self.allowed_roles)
        if principal.is_staff:
            ensure_assigned_hospital(principal)
        return principal


# ── Paged lists ──────────────────────────────────────────────────────────────

TOTAL_COUNT_HEADER = "X-Total-Count"


@dataclass(frozen=True)
class Page:
    """One page of a list. The full size goes in the X-Total-Count header."""

    limit: int
    offset: int

    def apply(self, query):
        return query.range(self.offset, self.offset + self.limit - 1)

    @staticmethod
    def report_total(response: Response, total: int | None) -> None:
        response.headers[TOTAL_COUNT_HEADER] = str(total or 0)


def page_params(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Page:
    return Page(limit=limit, offset=offset)


# ── Convenience role dependencies ────────────────────────────────────────────

require_nurse = RoleChecker([NURSE])
require_doctor = RoleChecker([DOCTOR])
require_admin = RoleChecker([ADMIN])
require_doctor_or_admin = RoleChecker([DOCTOR, ADMIN])
require_staff = RoleChecker([NURSE, DOCTOR, ADMIN])
require_government = RoleChecker([GOVERNMENT])
