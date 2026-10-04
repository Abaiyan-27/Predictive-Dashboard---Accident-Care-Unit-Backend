"""Transfer Pydantic schemas."""
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator
from uuid import UUID
from datetime import datetime

TransferStatus = Literal["pending_authorization", "authorized", "approved", "rejected"]


class TransferOut(BaseModel):
    transfer_id: UUID
    accident_id: Optional[UUID] = None
    from_hospital_id: Optional[UUID] = None
    to_hospital_id: Optional[UUID] = None
    status: Optional[str] = None
    request_message: Optional[str] = None
    requested_by: Optional[UUID] = None
    requested_at: Optional[datetime] = None
    authorized_by: Optional[UUID] = None
    authorization_reason: Optional[str] = None
    authorized_at: Optional[datetime] = None
    rejected_by: Optional[UUID] = None
    rejection_reason: Optional[str] = None
    processed_by: Optional[UUID] = None
    processed_at: Optional[datetime] = None
    processing_note: Optional[str] = None
    # Kept for rows written before the transfer workflow; new rows use processed_by.
    approved_by: Optional[UUID] = None
    decided_at: Optional[datetime] = None
    state_of_transfer: Optional[str] = None
    patient_name: Optional[str] = None
    source_hospital: Optional[str] = None
    destination_hospital: Optional[str] = None
    requested_by_name: Optional[str] = None
    authorized_by_name: Optional[str] = None
    rejected_by_name: Optional[str] = None
    processed_by_name: Optional[str] = None
    approved_by_name: Optional[str] = None
    allowed_actions: list[str] = Field(default_factory=list)


class CreateTransferPayload(BaseModel):
    accident_id: UUID
    to_hospital_id: UUID
    message: str = Field(min_length=1, max_length=2000)
    # Optional: the sending hospital is always the admin's own; if sent it must match.
    from_hospital_id: Optional[UUID] = None


class AuthorizeTransferPayload(BaseModel):
    note: Optional[str] = Field(default=None, max_length=2000)


class RejectTransferPayload(BaseModel):
    reason: str = Field(min_length=1, max_length=2000)


class ProcessTransferPayload(BaseModel):
    """The nurse's record of whether the physical transfer happened."""
    outcome: Literal["transferred", "not_transferred"]
    # How the patient was moved; required when transferred (a model input).
    state_of_transfer: Optional[str] = None
    note: Optional[str] = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _outcome_fields(self):
        if self.outcome == "transferred" and not self.state_of_transfer:
            raise ValueError("Record how the patient was transferred (state_of_transfer).")
        if self.outcome == "not_transferred" and not (self.note or "").strip():
            raise ValueError("Say why the transfer did not happen (note).")
        return self
