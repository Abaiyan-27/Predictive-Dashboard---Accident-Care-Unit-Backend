"""Socioeconomic Info Pydantic model — mirrors Supabase public.Socioeconomic_Info table."""
from pydantic import BaseModel
from typing import Optional
from uuid import UUID


class SocioeconomicInfo(BaseModel):
    patient_id: UUID
    family_income_before: Optional[str] = None
    family_income_after: Optional[str] = None
    family_current_status: Optional[str] = None
    insurance_claim_type: Optional[str] = None
