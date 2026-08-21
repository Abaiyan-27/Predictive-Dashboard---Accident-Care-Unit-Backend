"""Audit Log Pydantic model — mirrors Supabase public.AuditLog table."""
from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime


class AuditLog(BaseModel):
    log_id: UUID
    actor_id: UUID
    role: str
    hospital_id: Optional[UUID] = None
    action_type: str
    accident_id: Optional[UUID] = None
    timestamp: Optional[datetime] = None
