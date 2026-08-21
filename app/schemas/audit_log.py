"""Audit log Pydantic schemas."""
from typing import Optional
from pydantic import BaseModel
from uuid import UUID
from datetime import datetime


class AuditLogOut(BaseModel):
    log_id: UUID
    actor_id: UUID
    role: str
    hospital_id: Optional[UUID] = None
    action_type: str
    accident_id: Optional[UUID] = None
    timestamp: Optional[datetime] = None
