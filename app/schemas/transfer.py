"""Transfer Pydantic schemas."""
from typing import Literal, Optional
from pydantic import BaseModel
from uuid import UUID
from datetime import datetime

TransferStatus = Literal["pending_authorization", "authorized", "approved", "rejected"]


class TransferOut(BaseModel):
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
    status: Optional[str] = None
    patient_name: Optional[str] = None
    source_hospital: Optional[str] = None
    destination_hospital: Optional[str] = None


class CreateTransferPayload(BaseModel):
    accident_id: UUID
    from_hospital_id: UUID
    to_hospital_id: UUID
    authorization_reason: Optional[str] = None
    state_of_transfer: Optional[str] = None


class AuthorizeTransferPayload(BaseModel):
    authorization_reason: Optional[str] = None


class UpdateTransferStatus(BaseModel):
    status: Literal["authorized", "approved", "rejected"]
