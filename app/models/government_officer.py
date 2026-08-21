"""Government Officer Pydantic model — mirrors Supabase public.Government_Officer table."""
from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime


class GovernmentOfficer(BaseModel):
    officer_id: UUID
    name: str
    email: str
    nic: Optional[str] = None
    created_at: Optional[datetime] = None
