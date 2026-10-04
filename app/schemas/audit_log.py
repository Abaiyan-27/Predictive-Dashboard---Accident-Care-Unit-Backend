"""Audit log Pydantic schemas."""
from typing import Any, Optional
from pydantic import BaseModel
from uuid import UUID
from datetime import datetime


class AuditLogOut(BaseModel):
    log_id: UUID
    actor_id: UUID
    actor_name: Optional[str] = None
    role: str
    hospital_id: Optional[UUID] = None
    hospital_name: Optional[str] = None
    action_type: str
    accident_id: Optional[UUID] = None
    patient_name: Optional[str] = None
    timestamp: Optional[datetime] = None
    details: Optional[dict[str, Any]] = None
