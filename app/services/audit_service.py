"""Audit logging service — writes immutable audit records to the DB."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.audit_log import AuditLog


def log_action(
    db: Session,
    *,
    user_name: str,
    role: str,
    action: str,
) -> AuditLog:
    """
    Persist an audit log entry and flush (does NOT commit — caller owns the
    transaction so everything rolls back together on error).
    """
    entry = AuditLog(
        id=str(uuid.uuid4()),
        user=user_name,
        role=role,
        action=action,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    db.add(entry)
    db.flush()  # write to DB without committing
    return entry
