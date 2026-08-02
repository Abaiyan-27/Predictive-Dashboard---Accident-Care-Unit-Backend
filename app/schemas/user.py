"""User Pydantic schemas."""
from typing import Literal
from pydantic import BaseModel, EmailStr

Role = Literal["nurse", "doctor", "admin", "government"]


class UserOut(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    name: str
    email: str
    role: Role
    hospital_id: str
    hospital_name: str


class CreateUserPayload(BaseModel):
    name: str
    email: EmailStr
    role: Role
    hospital_id: str
    hospital_name: str = "Unknown Hospital"
