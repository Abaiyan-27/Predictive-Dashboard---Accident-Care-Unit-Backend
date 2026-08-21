"""User Pydantic model — mirrors Supabase public.User table."""
from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime


class User(BaseModel):
    user_id: UUID
    name: str
    email: str
    nic: Optional[str] = None
    role: str  # 'nurse' | 'doctor' | 'admin'
    hospital_id: Optional[UUID] = None
    created_at: Optional[datetime] = None
