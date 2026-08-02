"""Transfer Pydantic schemas."""
from typing import Literal
from pydantic import BaseModel

TransferStatus = Literal["pending", "approved", "rejected"]


class TransferOut(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    patient_name: str
    source_hospital: str
    destination_hospital: str
    status: TransferStatus
    requested_date: str


class UpdateTransferStatus(BaseModel):
    status: Literal["approved", "rejected"]
