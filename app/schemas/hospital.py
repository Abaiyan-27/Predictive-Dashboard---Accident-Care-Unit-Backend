"""Hospital Pydantic schemas."""
from pydantic import BaseModel


class HospitalStatus(BaseModel):
    model_config = {"from_attributes": True}

    total_beds: int
    occupied_beds: int
    available_beds: int
