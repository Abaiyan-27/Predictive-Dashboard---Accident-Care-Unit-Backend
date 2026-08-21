"""Hospital Patient Pydantic model — mirrors Supabase public.Hospital_Patient table."""
from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime


class HospitalPatient(BaseModel):
    id: UUID
    accident_id: Optional[UUID] = None
    hospital_id: Optional[UUID] = None
    entered_at: Optional[datetime] = None
    exited_at: Optional[datetime] = None
