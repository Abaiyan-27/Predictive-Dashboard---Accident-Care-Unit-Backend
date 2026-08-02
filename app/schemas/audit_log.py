"""Audit log Pydantic schemas."""
from typing import Literal
from pydantic import BaseModel


class AuditLogOut(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    user: str
    role: Literal["nurse", "doctor", "admin", "government"]
    action: str
    timestamp: str
