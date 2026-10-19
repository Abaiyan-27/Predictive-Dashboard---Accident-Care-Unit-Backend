"""Explainability: SHAP explanations of the doctor models' predictions.

See app/services/shap_explainer.py (ported from
Notebooks/Doctor_SHAP_Explainability.ipynb).
"""
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from supabase import Client

from app.api.deps import require_doctor
from app.core.authorization import ALL_MODELS, Principal, ensure_can_view_model, load_accident
from app.db.session import get_supabase
from app.services import doctor_workflow, shap_explainer
from app.services.shap_explainer import ExplainError, ModelLoadError, UnknownModelError

router = APIRouter(prefix="/xai", tags=["Explainability"])
logger = logging.getLogger(__name__)


class ExplainRequest(BaseModel):
    model_name: str
    accident_id: UUID


@router.post("/explain")
def get_explanation(
    request: ExplainRequest,
    current_user: Principal = Depends(require_doctor),
    supabase: Client = Depends(get_supabase),
):
    """Explain one doctor model's prediction for an episode with SHAP.

    Only doctors may ask, and only for episodes held by their own hospital. The
    explanation is computed from the same patient record the stored prediction
    was made from, and names that prediction when it is the current one.
    """
    try:
        model_key = shap_explainer.resolve_model(request.model_name)
    except UnknownModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if model_key in ALL_MODELS:
        ensure_can_view_model(current_user, model_key)
    accident = load_accident(supabase, current_user, request.accident_id, select=doctor_workflow.ACCIDENT_SELECT)

    try:
        explanation = shap_explainer.explain(model_key, doctor_workflow.model_record(accident))
    except ModelLoadError:
        logger.exception("SHAP explainer could not be loaded")
        raise HTTPException(status_code=503, detail="The explanation model is not available.")
    except ExplainError:
        logger.exception("SHAP explanation failed")
        raise HTTPException(status_code=422, detail="The explanation could not be generated for this patient.")
    except Exception:
        logger.exception("Unexpected SHAP error")
        raise HTTPException(status_code=500, detail="The explanation could not be generated.")

    # Link the explanation to the stored prediction the doctor is looking at.
    current = doctor_workflow.load_models_status(supabase, accident, role=current_user.role)[model_key]["prediction"]
    explanation["prediction_id"] = (
        current["prediction_id"]
        if current and not current.get("stale") and current.get("prediction") == explanation.get("prediction")
        else None
    )
    return explanation
