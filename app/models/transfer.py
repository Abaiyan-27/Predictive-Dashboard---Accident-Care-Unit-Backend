"""Transfer Pydantic model — mirrors Supabase public.Transfer table."""
from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime


class Transfer(BaseModel):
    transfer_id: UUID
    accident_id: Optional[UUID] = None
    from_hospital_id: Optional[UUID] = None
    to_hospital_id: Optional[UUID] = None
    authorized_by: Optional[UUID] = None
    authorization_reason: Optional[str] = None
    state_of_transfer: Optional[str] = None
    authorized_at: Optional[datetime] = None
    requested_by: Optional[UUID] = None
    requested_at: Optional[datetime] = None
    approved_by: Optional[UUID] = None
    decided_at: Optional[datetime] = None
    status: Optional[str] = "pending_authorization"
