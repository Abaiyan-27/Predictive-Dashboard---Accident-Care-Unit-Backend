"""The doctor's workflow: treatment, the three doctor models and their decisions.

Doctors decide Stay Duration, Transfer and Discharge independently. Each model
has its own stored predictions (append-only Model_Prediction rows) and at most
one final decision (a Prediction_Decision row referencing the exact prediction
the doctor saw). Severity belongs to the nurse; doctors only read it.

Treatment can be recorded until the first doctor-model decision. After that
it is locked, because a new treatment would change the inputs of predictions
that have already been decided.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from supabase import Client

from app.services import audit_service, ml_service, patient_service, prediction_store
from app.services.severity_workflow import FAILED_STATUSES, SCORED_STATUSES, latest_decision

DOCTOR_MODELS: tuple[str, ...] = ("stay_duration", "transfer", "discharge")

# Valid outputs and doctor decisions per model. Mirrors Model_Output_Label in
# the database and each model contract's label map (tests check both).
DECISION_LABELS: dict[str, tuple[str, ...]] = {
    "stay_duration": ("0-1 days", "2-3 days", "4+ days"),
    "transfer": ("Transfer", "No transfer"),
    "discharge": ("Complete recovery", "Partial recovery", "Further interventions"),
}
MODEL_TITLES = {
    "stay_duration": "Stay Duration",
    "transfer": "Transfer",
    "discharge": "Discharge Outcome",
}

# Per-model states, distinct from the Severity labels.
WAITING_FOR_INPUTS = "WAITING_FOR_INPUTS"      # treatment / required inputs missing
NOT_RUN = "NOT_RUN"                            # ready, no current prediction
PREDICTION_AVAILABLE = "PREDICTION_AVAILABLE"  # scored, awaiting the doctor
PREDICTION_FAILED = "PREDICTION_FAILED"        # artifact missing or inference error
ACCEPTED = "ACCEPTED"                          # doctor accepted the model output
OVERRIDDEN = "OVERRIDDEN"                      # doctor recorded their own decision

ACCIDENT_SELECT = "*, Patient(*), Injury_Description(*), Treatment(*), Transfer(*)"

PREDICTION_FIELDS = (
    "prediction_id,prediction_seq,accident_id,hospital_id,model_key,status,prediction,probability,"
    "probabilities,confidence_level,completeness,message,inputs_hash,trigger_event,requested_by,predicted_at"
)


def model_record(accident: dict[str, Any]) -> dict[str, Any]:
    patient = patient_service._first(accident.get("Patient"))
    return patient_service.build_model_record(patient=patient, accident=accident)


def current_inputs_hash(model_key: str, record: dict[str, Any]) -> str:
    return prediction_store.inputs_hash(prediction_store.model_inputs(model_key, record))


def treatments(accident: dict[str, Any]) -> list[dict[str, Any]]:
    rows = patient_service._list(accident.get("Treatment"))
    return sorted(rows, key=lambda row: str(row.get("created_at") or ""))


# ── Stored predictions and decisions ───────────────────────────────────────

def load_history(
    supabase: Client, accident_ids: Iterable[str]
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, list[dict[str, Any]]]]:
    """Doctor-model predictions per accident and model (newest first), and decisions per prediction."""
    ids = [str(accident_id) for accident_id in accident_ids]
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {
        accident_id: {model_key: [] for model_key in DOCTOR_MODELS} for accident_id in ids
    }
    decisions: dict[str, list[dict[str, Any]]] = {}
    if not ids:
        return predictions, decisions
    rows = (
        supabase.table("Model_Prediction")
        .select(PREDICTION_FIELDS)
        .in_("accident_id", ids)
        .in_("model_key", list(DOCTOR_MODELS))
        .order("prediction_seq", desc=True)
        .execute()
        .data
        or []
    )
    for row in rows:
        predictions.setdefault(str(row["accident_id"]), {}).setdefault(row["model_key"], []).append(row)
    if rows:
        decision_rows = (
            supabase.table("Prediction_Decision")
            .select("*")
            .in_("prediction_id", [row["prediction_id"] for row in rows])
            .order("decision_seq", desc=True)
            .execute()
            .data
            or []
        )
        for row in decision_rows:
            decisions.setdefault(str(row["prediction_id"]), []).append(row)
    return predictions, decisions


def _prediction_out(prediction: dict[str, Any], *, stale_reason: Optional[str]) -> dict[str, Any]:
    keys = (
        "prediction_id", "prediction_seq", "hospital_id", "status", "prediction", "probability",
        "probabilities", "confidence_level", "completeness", "message", "trigger_event",
        "requested_by", "predicted_at",
    )
    out = {key: prediction.get(key) for key in keys}
    out["stale"] = bool(stale_reason)
    out["stale_reason"] = stale_reason
    return out


def model_status(
    model_key: str,
    accident: dict[str, Any],
    record: dict[str, Any],
    predictions: list[dict[str, Any]],
    decisions: dict[str, list[dict[str, Any]]],
    *,
    role: Optional[str],
) -> dict[str, Any]:
    """The decision state of one doctor model, from stored rows only (no model run)."""
    readiness = ml_service.bundle_readiness(model_key, record)
    decided = latest_decision(predictions, decisions)
    prediction_out = None
    decision_out = None

    if decided:
        decision, decided_prediction = decided
        state = ACCEPTED if decision.get("decision_source") == "MODEL" else OVERRIDDEN
        prediction_out = _prediction_out(decided_prediction, stale_reason=None)
        decision_out = {
            "decision_id": decision.get("decision_id"),
            "prediction_id": decision.get("prediction_id"),
            "final_decision": decision.get("final_decision"),
            "decision_source": decision.get("decision_source"),
            "decided_by": decision.get("decided_by"),
            "decided_at": decision.get("decided_at"),
            "reason": decision.get("reason"),
            "model_prediction": decided_prediction.get("prediction"),
        }
    else:
        # A REFUSED row (stored by an earlier intake run) is the model saying
        # its inputs were missing: there is nothing to decide on.
        current = predictions[0] if predictions else None
        if current is not None and current.get("status") == "REFUSED":
            current = None
        stale_reason = None
        if current is not None:
            if str(current.get("hospital_id")) != str(accident.get("hospital_id")):
                stale_reason = "This prediction was made at another hospital; run the model again."
            elif current.get("inputs_hash") != current_inputs_hash(model_key, record):
                stale_reason = "The patient's inputs changed after this prediction; run the model again."
            prediction_out = _prediction_out(current, stale_reason=stale_reason)
        if current is None or stale_reason:
            state = NOT_RUN if readiness["ready"] else WAITING_FOR_INPUTS
        elif current.get("status") in SCORED_STATUSES:
            state = PREDICTION_AVAILABLE
        elif current.get("status") in FAILED_STATUSES:
            state = PREDICTION_FAILED
        else:
            state = NOT_RUN if readiness["ready"] else WAITING_FOR_INPUTS

    is_doctor = role == "doctor"
    return {
        "model_key": model_key,
        "title": MODEL_TITLES[model_key],
        "labels": list(DECISION_LABELS[model_key]),
        "state": state,
        "readiness": {
            "ready": bool(readiness["ready"]),
            "missing_required": readiness["missing_required"],
            "missing_features": readiness["missing_features"],
            "completeness": readiness["completeness"],
            "minimum_completeness": readiness["minimum_completeness"],
            "model_available": bool(readiness["available"]),
            "message": readiness["message"],
        },
        "prediction": prediction_out,
        "decision": decision_out,
        "final_decision": decision_out["final_decision"] if decision_out else None,
        "can_run": is_doctor and decided is None and bool(readiness["ready"]),
        "can_accept": is_doctor and state == PREDICTION_AVAILABLE,
        "can_override": is_doctor and state in {PREDICTION_AVAILABLE, PREDICTION_FAILED},
    }


def models_status(
    accident: dict[str, Any],
    predictions: dict[str, list[dict[str, Any]]],
    decisions: dict[str, list[dict[str, Any]]],
    *,
    role: Optional[str],
) -> dict[str, dict[str, Any]]:
    record = model_record(accident)
    return {
        model_key: model_status(
            model_key, accident, record, predictions.get(model_key, []), decisions, role=role
        )
        for model_key in DOCTOR_MODELS
    }


def treatment_status(accident: dict[str, Any], models: dict[str, dict[str, Any]], *, role: Optional[str]) -> dict[str, Any]:
    records = treatments(accident)
    locked = any(status["decision"] for status in models.values())
    return {
        "records": records,
        "recorded": bool(records),
        "locked": locked,
        "lock_reason": (
            "A doctor-model decision has been recorded; new treatment would change the inputs it was based on."
            if locked else None
        ),
        "can_record": role == "doctor" and not locked,
    }


def load_models_status(supabase: Client, accident: dict[str, Any], *, role: Optional[str]) -> dict[str, dict[str, Any]]:
    predictions, decisions = load_history(supabase, [accident["accident_id"]])
    return models_status(accident, predictions.get(str(accident["accident_id"]), {}), decisions, role=role)


def summaries_for(supabase: Client, accidents: list[dict[str, Any]], *, role: Optional[str]) -> dict[str, dict[str, Any]]:
    """Compact per-model state for a patient list (state, prediction label, final decision)."""
    predictions, decisions = load_history(supabase, [row["accident_id"] for row in accidents])
    out = {}
    for accident in accidents:
        models = models_status(accident, predictions.get(str(accident["accident_id"]), {}), decisions, role=role)
        out[str(accident["accident_id"])] = {
            "treatment_recorded": bool(treatments(accident)),
            "models": {
                key: {
                    "state": status["state"],
                    "prediction": (status["prediction"] or {}).get("prediction")
                    if status["prediction"] and not status["prediction"]["stale"] else None,
                    "final_decision": status["final_decision"],
                }
                for key, status in models.items()
            },
        }
    return out


# ── Running the doctor models ──────────────────────────────────────────────

def run_models(
    supabase: Client,
    accident: dict[str, Any],
    *,
    actor_id: str,
    role: str,
    trigger_event: str,
) -> dict[str, dict[str, Any]]:
    """Run every undecided doctor model whose inputs are complete and store the predictions.

    Models waiting for inputs are not run (no REFUSED rows are stored), and a
    model that already has a final decision is left alone. Returns the stored
    outputs by model key; an output without a prediction_id was not stored.
    """
    current = load_models_status(supabase, accident, role=role)
    runnable = tuple(
        key for key in DOCTOR_MODELS
        if current[key]["decision"] is None and current[key]["readiness"]["ready"]
    )
    if not runnable:
        return {}
    results = patient_service.predict_and_store(
        supabase,
        accident=accident,
        patient=patient_service._first(accident.get("Patient")),
        model_keys=runnable,
        trigger_event=trigger_event,
        requested_by=actor_id,
    )
    for model_key, output in results.items():
        audit_service.log_event(
            supabase,
            actor_id=actor_id,
            role=role,
            action_type=f"{model_key}_prediction_executed",
            hospital_id=accident.get("hospital_id"),
            accident_id=accident.get("accident_id"),
            details={
                "prediction_id": output.get("prediction_id"),
                "model": model_key,
                "model_prediction": output.get("prediction"),
                "model_status": output.get("status"),
                "probability": output.get("probability"),
                "trigger_event": trigger_event,
            },
        )
    return results
