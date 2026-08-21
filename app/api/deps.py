"""
Shared FastAPI dependencies
===========================
- get_current_user  — extracts + validates JWT, returns user dict
- RoleChecker       — callable dependency that enforces allowed roles
"""
from __future__ import annotations

from typing import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from supabase import Client

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


class RoleChecker:
    """Dependency factory that restricts access to specific roles."""

    def __init__(self, allowed_roles: list[str]) -> None:
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: dict = Depends(get_current_user)) -> dict:
        user_role = current_user.get("role", "")
        if user_role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {self.allowed_roles}",
            )
        return current_user


# ── Convenience role dependencies ────────────────────────────────────────────

require_nurse = RoleChecker(["nurse", "admin"])
require_doctor = RoleChecker(["doctor"])
require_doctor_or_admin = RoleChecker(["doctor", "admin"])
require_nurse_doctor_admin = RoleChecker(["nurse", "doctor", "admin"])
require_admin = RoleChecker(["admin"])
require_government = RoleChecker(["government"])
