"""Doctor workflow router — one episode's view, running the doctor models, decisions.

Stay Duration, Transfer and Discharge are separate decisions: each is recorded
against the exact stored prediction the doctor saw, as a new Prediction_Decision
row. The prediction row is never changed. Severity is shown read-only.
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from postgrest.exceptions import APIError
from supabase import Client

from app.api.deps import require_doctor
from app.core.authorization import Principal, ensure_can_decide, load_accident, load_patient, load_prediction
from app.db.session import get_supabase
from app.schemas.accident import ModelDecisionIn
from app.services import audit_service, doctor_workflow, patient_service, severity_workflow
from app.services.doctor_workflow import ACCIDENT_SELECT, DECISION_LABELS, DOCTOR_MODELS, SCORED_STATUSES

router = APIRouter(prefix="/doctor", tags=["Doctor / Decisions"])

BASIC_ACCIDENT_FIELDS = (
    "accident_id", "patient_id", "hospital_id", "created_at", "status", "incident_at",
    "collision_with", "vehicle_type", "mode_of_travel_during_accident", "mode_of_transport",
    "time_to_hospital", "helmet_worn", "road_type", "speed",
)


def episode_view(supabase: Client, principal: Principal, accident: dict) -> dict:
    """Everything the doctor's patient page shows, from stored rows only."""
    accident_id = str(accident["accident_id"])
    patient = patient_service._first(accident.get("Patient"))
    predictions, decisions = doctor_workflow.load_history(supabase, [accident_id])
    models = doctor_workflow.models_status(accident, predictions.get(accident_id, {}), decisions, role=principal.role)
    severity = severity_workflow.load_status(supabase, accident, role=principal.role)

    _, episodes = load_patient(
        supabase, principal, accident["patient_id"], "accident_id,created_at,status,hospital_id"
    )
    history = (
        supabase.table("Model_Prediction")
        .select(doctor_workflow.PREDICTION_FIELDS)
        .eq("accident_id", accident_id)
        .order("prediction_seq")
        .execute()
        .data
        or []
    )
    history_decisions = {}
    if history:
        for row in (
            supabase.table("Prediction_Decision")
            .select("*")
            .in_("prediction_id", [row["prediction_id"] for row in history])
            .order("decision_seq")
            .execute()
            .data
            or []
        ):
            history_decisions.setdefault(row["prediction_id"], []).append(row)
    return {
        "patient": {key: patient.get(key) for key in ("patient_id", "full_name", "age", "gender")},
        "accident": {
            **{key: accident.get(key) for key in BASIC_ACCIDENT_FIELDS},
            "injuries": patient_service.order_injuries(accident.get("Injury_Description")),
        },
        "severity": severity,
        "treatment": doctor_workflow.treatment_status(accident, models, role=principal.role),
        "models": models,
        "history": {
            "episodes": episodes,
            "transfers": patient_service._list(accident.get("Transfer")),
            "predictions": [
                {**row, "decisions": history_decisions.get(row["prediction_id"], [])} for row in history
            ],
        },
    }


@router.get("/accidents/{accident_id}")
def get_episode(
    accident_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_doctor),
):
    """The doctor's view of one episode held by their hospital."""
    accident = load_accident(supabase, current_user, accident_id, select=ACCIDENT_SELECT)
    return episode_view(supabase, current_user, accident)


@router.post("/accidents/{accident_id}/run")
def run_doctor_models(
    accident_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_doctor),
):
    """Run the undecided doctor models whose inputs are complete, and store the predictions."""
    accident = load_accident(supabase, current_user, accident_id, select=ACCIDENT_SELECT)
    models = doctor_workflow.load_models_status(supabase, accident, role=current_user.role)
    if all(models[key]["decision"] for key in DOCTOR_MODELS):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="All three doctor models have already been decided.")
    results = doctor_workflow.run_models(
        supabase, accident, actor_id=current_user.id, role=current_user.role, trigger_event="prognosis_viewed"
    )
    if not results:
        waiting = [
            f"{models[key]['title']} needs "
            + (", ".join(models[key]["readiness"]["missing_required"]) or "more of: " + ", ".join(models[key]["readiness"]["missing_features"]))
            for key in DOCTOR_MODELS if not models[key]["decision"]
        ]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No doctor model has the inputs it needs yet. " + "; ".join(waiting) + ".",
        )
    if any(not output.get("prediction_id") for output in results.values()):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="A prediction could not be saved. Please try again.")
    return episode_view(supabase, current_user, accident)


@router.post(
    "/accidents/{accident_id}/predictions/{prediction_id}/decision",
    status_code=status.HTTP_201_CREATED,
)
def decide(
    accident_id: UUID,
    prediction_id: UUID,
    body: ModelDecisionIn,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_doctor),
):
    """Record the doctor's final decision on one stored Stay, Transfer or Discharge prediction."""
    accident = load_accident(supabase, current_user, accident_id, select=ACCIDENT_SELECT)
    prediction = load_prediction(supabase, current_user, prediction_id)
    if str(prediction["accident_id"]) != str(accident_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found for this episode.")
    # Severity is the nurse's decision: a doctor is refused here (403).
    ensure_can_decide(current_user, prediction, accident)
    model_key = prediction["model_key"]
    if model_key not in DOCTOR_MODELS:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Doctors decide Stay Duration, Transfer and Discharge only.")

    labels = DECISION_LABELS[model_key]
    if body.final_decision is not None and body.final_decision not in labels:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{body.final_decision!r} is not a valid {model_key} decision. Choose one of: {', '.join(labels)}.",
        )

    predictions, decisions = doctor_workflow.load_history(supabase, [str(accident_id)])
    history = predictions.get(str(accident_id), {}).get(model_key, [])
    current = doctor_workflow.model_status(
        model_key, accident, doctor_workflow.model_record(accident), history, decisions, role=current_user.role
    )
    if current["decision"]:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"{current['title']} has already been decided for this patient.")
    if not history or str(history[0]["prediction_id"]) != str(prediction_id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"A newer {current['title']} prediction exists. Review it before deciding.")
    if current["prediction"] is None or current["prediction"]["stale"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(current["prediction"] or {}).get("stale_reason")
            or f"This {current['title']} prediction cannot be decided; run the model again.",
        )

    model_label = prediction.get("prediction")
    if body.action == "accept":
        if prediction.get("status") not in SCORED_STATUSES or model_label is None:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail="The model gave no prediction to accept. Record your own decision.")
        if body.final_decision is not None and body.final_decision != model_label:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Accepting records the model's prediction; to choose a different decision, override it.",
            )
        final_decision, source = model_label, "MODEL"
    else:
        if body.final_decision == model_label:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail="That is the model's prediction; accept it instead.")
        final_decision, source = body.final_decision, "DOCTOR"

    reason = (body.reason or "").strip() or None
    try:
        inserted = (
            supabase.table("Prediction_Decision")
            .insert({
                "prediction_id": str(prediction_id),
                "model_key": model_key,
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
        action_type=f"{model_key}_prediction_{'accepted' if source == 'MODEL' else 'overridden'}",
        hospital_id=accident.get("hospital_id"),
        accident_id=accident.get("accident_id"),
        details={
            "prediction_id": str(prediction_id),
            "decision_id": decision.get("decision_id"),
            "model": model_key,
            "model_prediction": model_label,
            "model_status": prediction.get("status"),
            "final_decision": final_decision,
            "decision_source": source,
            "decided_at": decision.get("decided_at"),
            "reason": reason,
        },
    )
    return episode_view(supabase, current_user, accident)
