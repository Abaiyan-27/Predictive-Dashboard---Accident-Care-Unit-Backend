"""The nurse's Severity workflow: which fields are editable, the current
Severity state of an episode, and recording the nurse's final decision.

Field groups (API field names, as used by the intake forms):

* basic details (name, gender) stay editable by nurses;
* Severity inputs (the model's twelve raw inputs, age included) are locked
  once a nurse has recorded the episode's Severity decision;
* inputs that only the later (doctor) models read are locked once a doctor has
  recorded treatment, because those models score them from then on.

The model output and the human decision stay separate: predictions are rows of
the append-only Model_Prediction table, and a decision is a new
Prediction_Decision row that references the exact prediction it was made on.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from supabase import Client

from app.contracts.severity import SEVERITY_DEFER
from app.services import ml_service, patient_service, prediction_store

SEVERITY = "severity"

# API field -> the severity model's raw input (build_model_record key).
SEVERITY_INPUT_FIELDS: dict[str, str] = {
    "incident_at": "Time of Collision",
    "age": "Age",
    "mode_of_travel_during_accident": "Mode Of Travel During Accident",
    "collision_with": "Collision With ",
    "vehicle_type": "Vehicle Type",
    "helmet_worn": "Helmet Worn",
    "passenger_type": "Passenger Type",
    "mode_of_transport": "Mode of Transport to the Hospital",
    "time_to_hospital": "Time Taken To Reach Hospital",
    "site_of_injury": "Site of Injury No1",
    "secondary_site_of_injury": "Site of injury No 2",
    "other_injury": "Other Injury",
}
_FIELD_FOR_RAW_INPUT = {raw: field for field, raw in SEVERITY_INPUT_FIELDS.items()}

# Accident details read only by the stay, transfer and discharge models.
LATER_MODEL_FIELDS: tuple[str, ...] = (
    "road_type",
    "speed",
    "road_condition",
    "visibility",
    "road_signals_exist",
    "collision_force_from",
    "alcohol_consumption",
    "illicit_drugs",
    "type_of_injury",
    "secondary_type_of_injury",
)
BASIC_FIELDS: tuple[str, ...] = ("full_name", "age", "gender")

# Severity states shown to staff. Each is distinct; none means "not classified".
NOT_RUN = "NOT_RUN"                            # enough inputs, model not run yet
WAITING_FOR_INPUTS = "WAITING_FOR_INPUTS"      # below the model's input minimum
PREDICTION_AVAILABLE = "PREDICTION_AVAILABLE"  # HIGH / BORDERLINE / LOW awaiting the nurse
MODEL_DEFERRED = "MODEL_DEFERRED"              # model declined to score (REFUSED)
PREDICTION_FAILED = "PREDICTION_FAILED"        # artifact missing or inference error
ACCEPTED = "ACCEPTED"                          # nurse accepted the model output
OVERRIDDEN = "OVERRIDDEN"                      # nurse recorded their own decision

SCORED_STATUSES = frozenset({"OK", "LOW_CONFIDENCE"})
FAILED_STATUSES = frozenset({"UNAVAILABLE", "ERROR"})

# Episode loads used by the workflow: everything build_model_record reads.
ACCIDENT_SELECT = "*, Patient(*), Injury_Description(*), Treatment(treatment_id)"
EMBEDDED_KEYS = ("Patient", "Injury_Description", "Treatment", "Transfer")

PREDICTION_FIELDS = (
    "prediction_id,prediction_seq,accident_id,hospital_id,model_key,model_version_id,"
    "status,prediction,probability,probabilities,confidence_level,completeness,message,"
    "inputs_hash,trigger_event,requested_by,predicted_at"
)


# ── Inputs ─────────────────────────────────────────────────────────────────

def model_record(accident: dict[str, Any]) -> dict[str, Any]:
    patient = patient_service._first(accident.get("Patient"))
    return patient_service.build_model_record(patient=patient, accident=accident)


def current_inputs_hash(record: dict[str, Any]) -> str:
    """Hash of the severity inputs as stored now, comparable with Model_Prediction.inputs_hash."""
    return prediction_store.inputs_hash(prediction_store.model_inputs(SEVERITY, record))


def input_summary(record: dict[str, Any]) -> dict[str, Any]:
    readiness = ml_service.severity_readiness(record)
    if readiness["available"]:
        present = {_FIELD_FOR_RAW_INPUT.get(raw, raw) for raw in readiness["present"]}
        total = len(readiness["raw_inputs"])
    else:
        present = {
            field for field, raw in SEVERITY_INPUT_FIELDS.items()
            if not ml_service._missing(ml_service._normalise_value(record.get(raw)))
        }
        total = len(SEVERITY_INPUT_FIELDS)
    return {
        "recorded": len(present),
        "total": total,
        "minimum": readiness["minimum"],
        "complete": len(present) >= total,
        "ready": bool(readiness["ready"]),
        "missing": [field for field in SEVERITY_INPUT_FIELDS if field not in present],
        "model_available": bool(readiness["available"]),
        "model_message": readiness["message"],
    }


def basic_complete(accident: dict[str, Any]) -> bool:
    patient = patient_service._first(accident.get("Patient"))
    return all(str(patient.get(field) or "").strip() for field in BASIC_FIELDS)


# ── Stored predictions and decisions ───────────────────────────────────────

def load_history(
    supabase: Client, accident_ids: Iterable[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Severity predictions per accident (newest first) and decisions per prediction (newest first)."""
    ids = [str(accident_id) for accident_id in accident_ids]
    predictions: dict[str, list[dict[str, Any]]] = {accident_id: [] for accident_id in ids}
    decisions: dict[str, list[dict[str, Any]]] = {}
    if not ids:
        return predictions, decisions
    rows = (
        supabase.table("Model_Prediction")
        .select(PREDICTION_FIELDS)
        .in_("accident_id", ids)
        .eq("model_key", SEVERITY)
        .order("prediction_seq", desc=True)
        .execute()
        .data
        or []
    )
    for row in rows:
        predictions.setdefault(str(row["accident_id"]), []).append(row)
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


def latest_decision(
    predictions: list[dict[str, Any]], decisions: dict[str, list[dict[str, Any]]]
) -> Optional[tuple[dict[str, Any], dict[str, Any]]]:
    """The episode's most recent Severity decision and the prediction it was made on."""
    found = [
        (decision, prediction)
        for prediction in predictions
        for decision in decisions.get(str(prediction["prediction_id"]), [])
    ]
    if not found:
        return None
    return max(found, key=lambda pair: pair[0].get("decision_seq") or 0)


def _prediction_out(prediction: dict[str, Any], *, stale: bool, stale_reason: Optional[str]) -> dict[str, Any]:
    keys = (
        "prediction_id", "prediction_seq", "hospital_id", "status", "prediction", "probability",
        "probabilities", "confidence_level", "completeness", "message", "trigger_event",
        "requested_by", "predicted_at",
    )
    out = {key: prediction.get(key) for key in keys}
    out["is_defer"] = prediction.get("prediction") == SEVERITY_DEFER
    out["stale"] = stale
    out["stale_reason"] = stale_reason
    return out


def episode_status(
    accident: dict[str, Any],
    predictions: list[dict[str, Any]],
    decisions: dict[str, list[dict[str, Any]]],
    *,
    role: Optional[str] = None,
) -> dict[str, Any]:
    """The Severity workflow state of one episode, from stored rows only (no model run)."""
    record = model_record(accident)
    inputs = input_summary(record)
    decided = latest_decision(predictions, decisions)
    later_locked = bool(patient_service._list(accident.get("Treatment")))

    prediction_out = None
    decision_out = None
    if decided:
        decision, decided_prediction = decided
        state = ACCEPTED if decision.get("decision_source") == "MODEL" else OVERRIDDEN
        prediction_out = _prediction_out(decided_prediction, stale=False, stale_reason=None)
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
        current = predictions[0] if predictions else None
        stale_reason = None
        if current is not None:
            if str(current.get("hospital_id")) != str(accident.get("hospital_id")):
                stale_reason = "This prediction was made at another hospital; run Severity again."
            elif current.get("inputs_hash") != current_inputs_hash(record):
                stale_reason = "Severity inputs changed after this prediction; run Severity again."
            prediction_out = _prediction_out(current, stale=bool(stale_reason), stale_reason=stale_reason)
        if current is None or stale_reason:
            state = NOT_RUN if inputs["ready"] else WAITING_FOR_INPUTS
        elif current.get("status") in SCORED_STATUSES:
            state = PREDICTION_AVAILABLE
        elif current.get("status") in FAILED_STATUSES:
            state = PREDICTION_FAILED
        else:
            state = MODEL_DEFERRED

    severity_locked = decided is not None
    locked = (list(SEVERITY_INPUT_FIELDS) if severity_locked else []) + (
        list(LATER_MODEL_FIELDS) if later_locked else []
    )
    is_nurse = role == "nurse"
    decidable = state in {PREDICTION_AVAILABLE, MODEL_DEFERRED, PREDICTION_FAILED}
    return {
        "accident_id": str(accident.get("accident_id")),
        "state": state,
        "basic_complete": basic_complete(accident),
        "inputs": inputs,
        "prediction": prediction_out,
        "decision": decision_out,
        "final_severity": decision_out["final_decision"] if decision_out else None,
        "locks": {
            "severity_inputs": severity_locked,
            "later_model_inputs": later_locked,
            "fields": locked,
        },
        "can_run": is_nurse and not severity_locked and inputs["ready"],
        "can_accept": is_nurse and state == PREDICTION_AVAILABLE,
        "can_override": is_nurse and decidable,
    }


def load_status(supabase: Client, accident: dict[str, Any], *, role: Optional[str]) -> dict[str, Any]:
    predictions, decisions = load_history(supabase, [accident["accident_id"]])
    return episode_status(
        accident, predictions.get(str(accident["accident_id"]), []), decisions, role=role
    )


def statuses_for(
    supabase: Client, accidents: list[dict[str, Any]], *, role: Optional[str]
) -> dict[str, dict[str, Any]]:
    predictions, decisions = load_history(supabase, [row["accident_id"] for row in accidents])
    return {
        str(row["accident_id"]): episode_status(
            row, predictions.get(str(row["accident_id"]), []), decisions, role=role
        )
        for row in accidents
    }


# ── Edit locks ─────────────────────────────────────────────────────────────

def _as_datetime(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _same(field: str, stored: Any, submitted: Any) -> bool:
    if field == "incident_at":
        return _as_datetime(stored) == _as_datetime(submitted)
    return str(stored if stored is not None else "") == str(submitted if submitted is not None else "")


def _stored_values(accident: dict[str, Any]) -> dict[str, Any]:
    patient = patient_service._first(accident.get("Patient"))
    first, second = patient_service._injury_slots(accident.get("Injury_Description"))
    values = {field: accident.get(field) for field in SEVERITY_INPUT_FIELDS}
    values.update({field: accident.get(field) for field in LATER_MODEL_FIELDS})
    values.update({
        "full_name": patient.get("full_name"),
        "age": patient.get("age"),
        "gender": patient.get("gender"),
        "site_of_injury": first.get("site_of_injury"),
        "type_of_injury": first.get("type_of_injury"),
        "secondary_site_of_injury": second.get("site_of_injury"),
        "secondary_type_of_injury": second.get("type_of_injury"),
    })
    return values


def _submitted_values(form: Any) -> dict[str, Any]:
    """The values an UpdateIntakeForm would write. None means "leave unchanged"."""
    submitted = {
        field: getattr(form, field, None)
        for field in (*SEVERITY_INPUT_FIELDS, *LATER_MODEL_FIELDS, "full_name", "gender")
        if hasattr(form, field)
    }
    if getattr(form, "injuries", None):
        # A non-empty injury list replaces both slots: an omitted slot is removed.
        slots = {injury.injury_order: injury for injury in form.injuries}
        for order, site, kind in (
            (1, "site_of_injury", "type_of_injury"),
            (2, "secondary_site_of_injury", "secondary_type_of_injury"),
        ):
            injury = slots.get(order)
            submitted[site] = injury.site_of_injury if injury else ""
            submitted[kind] = injury.type_of_injury if injury else ""
    return submitted


def lock_violations(form: Any, accident: dict[str, Any], status: dict[str, Any]) -> list[str]:
    """Locked fields the form would change. Re-sending the stored value is allowed."""
    locked = set(status["locks"]["fields"])
    stored = _stored_values(accident)
    injuries_sent = bool(getattr(form, "injuries", None))
    violations = []
    for field, value in _submitted_values(form).items():
        if field not in locked:
            continue
        injury_field = field in {"site_of_injury", "type_of_injury", "secondary_site_of_injury", "secondary_type_of_injury"}
        if value is None and not (injury_field and injuries_sent):
            continue
        if not _same(field, stored.get(field), value):
            violations.append(field)
    return violations
