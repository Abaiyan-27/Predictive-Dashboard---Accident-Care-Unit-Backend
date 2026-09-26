from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from supabase import Client

from app.services.xai_service import explain, UnknownModelError, ModelLoadError, XAIEngineError
from app.api.deps import get_current_user
from app.db.session import get_supabase
from app.services import patient_service

router = APIRouter(prefix="/xai", tags=["Explainability"])

class ExplainRequest(BaseModel):
    model_name: str
    accident_id: str

@router.post("/explain")
def get_explanation(
    request: ExplainRequest, 
    current_user: Any = Depends(get_current_user),
    supabase: Client = Depends(get_supabase)
):
    """
    Explain a model prediction using Value of Information and Counterfactual sensitivity.
    Requires an authenticated user (doctor, nurse, or admin).
    """
    try:
        # Fetch the necessary records
        accidents = (
            supabase.table("Accident_Record")
            .select("*, Injury_Description(*), Treatment(*)")
            .eq("accident_id", request.accident_id)
            .limit(1)
            .execute()
        )
        if not accidents.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No accident record found.")
        accident = accidents.data[0]
        
        patient_id = accident.get("patient_id")
        patient = {}
        if patient_id:
            patient_result = supabase.table("Patient").select("*").eq("patient_id", patient_id).execute()
            if patient_result.data:
                patient = patient_result.data[0]
                
        transfer_result = (
            supabase.table("Transfer")
            .select("*")
            .eq("accident_id", request.accident_id)
            .order("requested_at", desc=True)
            .execute()
        )
        
        model_record = patient_service.build_discharge_record(
            patient=patient,
            accident=accident,
            transfers=transfer_result.data or [],
        )

        explanation = explain(request.model_name, model_record)
        return explanation
    except UnknownModelError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelLoadError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except XAIEngineError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
