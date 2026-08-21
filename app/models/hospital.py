"""Hospital Pydantic model — mirrors Supabase public.Hospital table."""
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID


class Hospital(BaseModel):
    hospital_id: UUID
    name: str
    type: Optional[str] = None
    region: Optional[str] = None
    district: Optional[str] = None
    address: Optional[str] = None
