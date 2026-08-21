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
    password: str = Field(min_length=8)
    nic: Optional[str] = None
    role: StaffRole
    hospital_id: Optional[UUID] = None
