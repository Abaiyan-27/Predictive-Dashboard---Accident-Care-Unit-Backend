"""Audit logging service — writes immutable audit records to Supabase."""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from supabase import Client


def log_action(
    supabase: Client,
    *,
    actor_id: UUID,
    role: str,
    action_type: str,
    hospital_id: Optional[UUID] = None,
    accident_id: Optional[UUID] = None,
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

    result = supabase.table("AuditLog").insert(entry).execute()
    return result.data[0] if result.data else {}
