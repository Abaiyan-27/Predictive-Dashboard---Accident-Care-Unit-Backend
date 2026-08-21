"""Treatment Pydantic model — mirrors Supabase public.Treatment table."""
from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime


class Treatment(BaseModel):
    treatment_id: UUID
    accident_id: Optional[UUID] = None
    hospital_id: Optional[UUID] = None
    ward_number: Optional[str] = None
    reason: Optional[str] = None
    number_of_days_stay: Optional[int] = None
    decided_by: Optional[UUID] = None
    logged_by: Optional[UUID] = None
    created_at: Optional[datetime] = None
