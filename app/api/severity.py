"""Severity router — the nurse runs the Severity model and records the final decision.

The model's output is stored as a Model_Prediction row and is never changed.
The nurse then either accepts it (decision_source MODEL) or records their own
decision (decision_source NURSE) as a separate Prediction_Decision row that
references the exact prediction she was shown.
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from postgrest.exceptions import APIError
from supabase import Client

from app.api.deps import require_nurse, require_staff
from app.core.authorization import Principal, ensure_can_decide, load_accident, load_prediction
from app.db.session import get_supabase
from app.schemas.accident import SeverityDecisionIn
from app.services import audit_service, patient_service, severity_workflow
from app.services.severity_workflow import ACCIDENT_SELECT, SCORED_STATUSES, SEVERITY

router = APIRouter(prefix="/severity", tags=["Nurse / Severity"])


@router.get("/accidents/{accident_id}")
def get_severity_status(
    accident_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_staff),
):
    """Current Severity state of an episode: inputs, latest prediction, decision and locks."""
    accident = load_accident(supabase, current_user, accident_id, select=ACCIDENT_SELECT)
    return severity_workflow.load_status(supabase, accident, role=current_user.role)


@router.post("/accidents/{accident_id}/run")
def run_severity(
    accident_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_nurse),
):
    """Run the real Severity model on the episode's saved inputs and store the prediction."""
    accident = load_accident(supabase, current_user, accident_id, select=ACCIDENT_SELECT)
    current = severity_workflow.load_status(supabase, accident, role=current_user.role)
    if current["locks"]["severity_inputs"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Severity has already been decided for this patient.",
        )
    inputs = current["inputs"]
    if not inputs["ready"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Severity needs at least {inputs['minimum']} of {inputs['total']} inputs; "
                f"{inputs['recorded']} recorded. Missing: {', '.join(inputs['missing'])}."
            ),
        )

    results = patient_service.predict_and_store(
        supabase,
        accident=accident,
        patient=patient_service._first(accident.get("Patient")),
        model_keys=(SEVERITY,),
        trigger_event="intake_updated",
        requested_by=current_user.id,
    )
    output = results[SEVERITY]
    if not output.get("prediction_id"):
        # A result that is not stored cannot be decided on, so it is not shown.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The Severity prediction could not be saved. Please try again.",
        )

    audit_service.log_event(
        supabase,
        actor_id=current_user.id,
        role=current_user.role,
        action_type="severity_prediction_executed",
        hospital_id=accident.get("hospital_id"),
        accident_id=accident.get("accident_id"),
        details={
            "prediction_id": output["prediction_id"],
            "model_prediction": output.get("prediction"),
            "model_status": output.get("status"),
            "probability": output.get("probability"),
            "completeness": output.get("completeness"),
        },
    )
    return severity_workflow.load_status(supabase, accident, role=current_user.role)


@router.post("/predictions/{prediction_id}/decision", status_code=status.HTTP_201_CREATED)
def decide_severity(
    prediction_id: UUID,
    body: SeverityDecisionIn,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_nurse),
):
    """Record the nurse's final Severity decision on one stored prediction."""
    # 404 for a missing prediction, one at another hospital, or one the role cannot see.
    prediction = load_prediction(supabase, current_user, prediction_id)
    if prediction["model_key"] != SEVERITY:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Nurses record decisions on Severity predictions only.",
        )
    accident = load_accident(supabase, current_user, prediction["accident_id"], select=ACCIDENT_SELECT)
    ensure_can_decide(current_user, prediction, accident)

    predictions, decisions = severity_workflow.load_history(supabase, [accident["accident_id"]])
    history = predictions.get(str(accident["accident_id"]), [])
    if severity_workflow.latest_decision(history, decisions):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Severity has already been decided for this patient.",
        )
    if not history or str(history[0]["prediction_id"]) != str(prediction_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A newer Severity prediction exists for this patient. Review it before deciding.",
        )
    current = severity_workflow.episode_status(accident, history, decisions, role=current_user.role)
    if current["prediction"]["stale"]:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=current["prediction"]["stale_reason"])

    model_label = prediction.get("prediction")
    if body.action == "accept":
        if prediction.get("status") not in SCORED_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "The model deferred to clinical judgement, so there is no model decision to "
                    "accept. Choose HIGH, BORDERLINE or LOW as your own decision."
                ),
            )
        final_decision, source = model_label, "MODEL"
    else:
        if body.final_decision == model_label:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="That is the model's prediction; accept it instead.",
            )
        final_decision, source = body.final_decision, "NURSE"

    reason = (body.reason or "").strip() or None
    try:
        inserted = (
            supabase.table("Prediction_Decision")
            .insert({
                "prediction_id": str(prediction_id),
                "model_key": SEVERITY,
                "final_decision": final_decision,
                "decision_source": source,
                "decided_by": current_user.id,
                "reason": reason,
            })
            .execute()
            .data
        )
    except APIError as exc:
        # The database re-checks role, hospital and label rules on every insert.
        # 23505: another decision for this patient and model won a concurrent race.
        if exc.code in {"23514", "23503", "23505"}:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=exc.message) from exc
        raise
    if not inserted:
        raise HTTPException(status_code=500, detail="Failed to record the decision.")
    decision = inserted[0]

    audit_service.log_event(
        supabase,
        actor_id=current_user.id,
        role=current_user.role,
        action_type=(
            "severity_prediction_accepted" if source == "MODEL" else "severity_prediction_overridden"
        ),
        hospital_id=accident.get("hospital_id"),
        accident_id=accident.get("accident_id"),
        details={
            "prediction_id": str(prediction_id),
            "decision_id": decision.get("decision_id"),
            "model_prediction": model_label,
            "model_status": prediction.get("status"),
            "final_decision": final_decision,
            "decision_source": source,
            "decided_at": decision.get("decided_at"),
            "reason": reason,
        },
    )
    return severity_workflow.load_status(supabase, accident, role=current_user.role)
