import logging
from typing import Dict
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from supabase import Client

from app.services.xai_service import explain, UnknownModelError, ModelLoadError, XAIEngineError
from app.api.deps import require_doctor
from app.core.authorization import ALL_MODELS, Principal, ensure_can_view_model, load_accident
from app.db.session import get_supabase
from app.services import patient_service

router = APIRouter(prefix="/xai", tags=["Explainability"])
logger = logging.getLogger(__name__)

class ExplainRequest(BaseModel):
    model_name: str
    accident_id: UUID

@router.post("/explain")
def get_explanation(
    request: ExplainRequest, 
    current_user: Principal = Depends(require_doctor),
    supabase: Client = Depends(get_supabase)
):
    """
    Explain a model prediction using Value of Information and Counterfactual sensitivity.
    Explanations cover the doctor models, so only doctors may request them,
    and only for episodes held by their own hospital.
    """
    try:
        if request.model_name in ALL_MODELS:
            ensure_can_view_model(current_user, request.model_name)
        accident = load_accident(
            supabase, current_user, request.accident_id, "*, Injury_Description(*), Treatment(*)"
        )
        
        patient_id = accident.get("patient_id")
        patient = {}
        if patient_id:
            patient_result = supabase.table("Patient").select("*").eq("patient_id", patient_id).execute()
            if patient_result.data:
                patient = patient_result.data[0]
                
        transfer_result = (
            supabase.table("Transfer")
            .select("*")
            .eq("accident_id", str(request.accident_id))
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
    except ModelLoadError:
        logger.exception("XAI model could not be loaded")
        raise HTTPException(status_code=503, detail="The explanation model is not available.")
    except XAIEngineError:
        logger.exception("XAI explanation failed")
        raise HTTPException(status_code=422, detail="The explanation could not be generated for this patient.")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected XAI error")
        raise HTTPException(status_code=500, detail="The explanation could not be generated.")
