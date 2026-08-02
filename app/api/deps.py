"""
Shared FastAPI dependencies
===========================
- get_current_user  — extracts + validates JWT, returns User ORM object
- RoleChecker       — callable dependency that enforces allowed roles
"""
from __future__ import annotations

from typing import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy.orm import Session

from app.core.security import decode_token
from app.db.session import get_db
from app.models.user import User

bearer_scheme = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Validate JWT and return the authenticated User ORM object."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(credentials.credentials)
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.get(User, user_id)
    if user is None:
        raise credentials_exception
    return user


class RoleChecker:
    """Dependency factory that restricts access to specific roles."""

    def __init__(self, allowed_roles: list[str]) -> None:
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in self.allowed_roles:
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
