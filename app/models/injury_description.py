"""Injury Description Pydantic model — mirrors Supabase public.Injury_Description table."""
from pydantic import BaseModel
from typing import Optional
from uuid import UUID


class InjuryDescription(BaseModel):
    injury_id: UUID
    accident_id: Optional[UUID] = None
    site_of_injury: Optional[str] = None
    type_of_injury: Optional[str] = None
