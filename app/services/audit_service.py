"""Audit logging service — writes immutable audit records to Supabase."""
from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID

from supabase import Client

logger = logging.getLogger(__name__)


def log_action(
    supabase: Client,
    *,
    actor_id: UUID,
    role: str,
    action_type: str,
    hospital_id: Optional[UUID] = None,
    accident_id: Optional[UUID] = None,
    details: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Persist an audit log entry to the AuditLog table in Supabase.
    Timestamp is auto-set by Supabase (DEFAULT now()).
    """
    entry = {
        "actor_id": str(actor_id),
        "role": role,
        "action_type": action_type,
    }
    if hospital_id:
        entry["hospital_id"] = str(hospital_id)
    if accident_id:
        entry["accident_id"] = str(accident_id)
    if details:
        entry["details"] = details

    result = supabase.table("AuditLog").insert(entry).execute()
    return result.data[0] if result.data else {}


def log_event(supabase: Client, **kwargs: Any) -> dict:
    """Like log_action, but an audit failure never fails the caller's request.

    If the database has not yet received the AuditLog.details column, the entry
    is written without details rather than dropped.
    """
    try:
        return log_action(supabase, **kwargs)
    except Exception:
        if kwargs.get("details"):
            try:
                logger.warning("Audit details not stored for %s; retrying without them", kwargs.get("action_type"))
                return log_action(supabase, **{**kwargs, "details": None})
            except Exception:
                pass
        logger.exception("Audit log entry %s could not be written", kwargs.get("action_type"))
        return {}
