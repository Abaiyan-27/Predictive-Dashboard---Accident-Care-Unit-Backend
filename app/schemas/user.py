"""User Pydantic schemas."""
from typing import Literal, Optional
from pydantic import BaseModel, EmailStr, Field
from uuid import UUID
from datetime import datetime

Role = Literal["nurse", "doctor", "admin", "government"]
StaffRole = Literal["nurse", "doctor", "admin"]


class UserOut(BaseModel):
    user_id: UUID
    name: str
    email: str
    nic: Optional[str] = None
    role: Role
    hospital_id: Optional[UUID] = None
    hospital_name: Optional[str] = None
    created_at: Optional[datetime] = None


class CreateUserPayload(BaseModel):
    name: str
    email: EmailStr
    # Omit to have a temporary password generated. Either way the new user must
    # replace it at first sign-in.
    password: Optional[str] = Field(default=None, min_length=8)
    nic: Optional[str] = None
    role: StaffRole
    hospital_id: Optional[UUID] = None


class CreatedUserOut(UserOut):
    # Shown once to the creating admin when generated; never stored by the API.
    temporary_password: Optional[str] = None
    must_change_password: bool = True
