"""Helpers for enriching Supabase accident records with real model outputs."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from app.schemas.accident import SeverityPredictionOut
from supabase import Client

from app.services import ml_service, prediction_store


def _first(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        return value[0] if value else {}
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return [value] if isinstance(value, dict) else []


def order_injuries(injuries: Any) -> list[dict[str, Any]]:
    """Return injuries sorted by injury_order (1 = primary, 2 = secondary)."""
    rows = _list(injuries)
    return sorted(rows, key=lambda row: (row.get("injury_order") is None, row.get("injury_order") or 0))


def _injury_slots(injuries: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pick the primary and secondary injury by their stored order, never by row position."""
    rows = _list(injuries)
    if any("injury_order" in row for row in rows):
        by_order = {row.get("injury_order"): row for row in rows}
        return by_order.get(1, {}), by_order.get(2, {})
    # Rows read before the injury_order migration carry no order at all.
    return (rows[0] if rows else {}), (rows[1] if len(rows) > 1 else {})


def _collision_time_bucket(value: Any) -> Any:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    labels = [
        "00:00 - 03:00",
        "03:00 - 06:00",
        "06:00 - 09:00",
        "09:00 - 12:00",
        "12:00 -15:00",
        "15:00 - 18:00",
        "18:00 - 21:00",
        "21:00 - 00:00",
    ]
    return labels[parsed.hour // 3]


def _age_bucket(value: Any) -> Any:
    if value is None or str(value).strip() == "":
        return None
    try:
        val = int(value)
    except (ValueError, TypeError):
        return value
    val = max(1, min(100, val))
    if val <= 9:
        return "1-9"
    lower = min((val // 10) * 10, 90)
    upper = 100 if lower == 90 else lower + 9
    return f"{lower}-{upper}"


# The source dataset records every patient who was not transferred with this
# exact (misspelt) State of Transfer value, so the notebooks were trained on it.
NOT_TRANSFERRED_STATE = "tranfer without any support"
OPEN_TRANSFER_STATUSES = {"pending_authorization", "authorized"}


def _transfer_inputs(
    transfers: list[dict[str, Any]],
    treatment_recorded: bool,
) -> tuple[Any, Any]:
    """Return (Transfer To Next Hospital, State of Transfer) from recorded transfers.

    A completed transfer is a recorded fact. "Not transferred" is only stated
    once a doctor has recorded treatment and no transfer is still open;
    before that the outcome is unknown and both inputs stay missing.
    """
    approved = [item for item in transfers if item.get("status") == "approved"]
    if approved:
        latest = max(
            approved,
            key=lambda item: str(item.get("decided_at") or item.get("requested_at") or ""),
        )
        return "yes", latest.get("state_of_transfer")
    if treatment_recorded and not any(
        item.get("status") in OPEN_TRANSFER_STATUSES for item in transfers
    ):
        return "no", NOT_TRANSFERRED_STATE
    return None, None


def build_model_record(
    *,
    patient: dict[str, Any],
    accident: dict[str, Any],
    transfers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Map database rows to the notebooks' raw column names.

    Only values that were actually recorded are passed through. Missing
    values stay None so each model's completeness reflects what staff entered.
    """
    first_injury, second_injury = _injury_slots(accident.get("Injury_Description"))
    treatments = _list(accident.get("Treatment"))
    known_transfers = transfers if transfers is not None else _list(accident.get("Transfer"))
    latest_treatment = max(
        treatments,
        key=lambda treatment: str(treatment.get("created_at") or ""),
    ) if treatments else {}
    recorded_days = [
        int(treatment["number_of_days_stay"])
        for treatment in treatments
        if treatment.get("number_of_days_stay") is not None
    ]
    total_days = sum(recorded_days) if recorded_days else None
    transferred, transfer_state = _transfer_inputs(known_transfers, bool(treatments))

    return {
        # ── Severity-triage raw inputs (12 fields) ────────────────────────
        "Time of Collision": _collision_time_bucket(accident.get("incident_at")),
        "Age": _age_bucket(patient.get("age")),
        "Mode Of Travel During Accident": accident.get("mode_of_travel_during_accident"),
        "Collision With ": accident.get("collision_with"),
        "Vehicle Type": accident.get("vehicle_type"),
        "Helmet Worn": accident.get("helmet_worn"),
        "Passenger Type": accident.get("passenger_type"),
        "Mode of Transport to the Hospital": accident.get("mode_of_transport"),
        "Time Taken To Reach Hospital": accident.get("time_to_hospital"),
        "Site of Injury No1": first_injury.get("site_of_injury"),
        "Site of injury No 2": second_injury.get("site_of_injury"),
        "Other Injury": accident.get("other_injury"),
        # ── Post-treatment discharge / stay / transfer inputs ─────────────
        "Collision With": accident.get("collision_with"),
        "Approximate Speed": accident.get("speed"),
        "Road Type": accident.get("road_type"),
        "Road Condition": accident.get("road_condition"),
        "Visiblity": accident.get("visibility"),
        "Road Signals Exist": accident.get("road_signals_exist"),
        "Collision Force From": accident.get("collision_force_from"),
        "Alcohol Consumption": accident.get("alcohol_consumption"),
        "Illicit Drugs": accident.get("illicit_drugs"),
        "Type of injury No 1": first_injury.get("type_of_injury"),
        "Type of Injury No 2": second_injury.get("type_of_injury"),
        "Transfer To Next Hospital": transferred,
        "State of Transfer": transfer_state,
        "Investigation Done": latest_treatment.get("investigation_done"),
        "Management Name 1": latest_treatment.get("management_name_1"),
        "Management Name 2": latest_treatment.get("management_name_2"),
        "total_days": total_days,
    }


INTAKE_MODELS = ("severity", "stay_duration", "transfer")
ALL_MODELS = ("severity", "stay_duration", "transfer", "discharge")
RESPONSE_KEYS = {
    "severity": "severity_prediction",
    "stay_duration": "stay_duration_prediction",
    "transfer": "transfer_prediction",
    "discharge": "discharge_prediction",
}
_RUNNERS = {
    "severity": lambda record: ml_service.predict_severity(record=record),
    "stay_duration": ml_service.predict_stay_duration,
    "transfer": ml_service.predict_transfer_probability,
    "discharge": ml_service.predict_discharge_outcome,
}


def run_models(record: dict[str, Any], model_keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for key in model_keys:
        output = _RUNNERS[key](record).to_dict()
        if key == "severity":
            # Validating against the shared contract means an unexpected label
            # fails loudly here instead of reaching a dashboard.
            output = SeverityPredictionOut(**output).model_dump()
        results[key] = output
    return results


def add_patient_predictions(accident: dict[str, Any]) -> dict[str, Any]:
    """Attach intake-time predictions for display. Does not persist anything.

    Each output is marked ``persisted: False`` with no ``prediction_id``: it is
    computed on read, is not part of the prediction history, and can never be
    accepted or overridden (decisions reference stored predictions only).
    """
    accident["Injury_Description"] = order_injuries(accident.get("Injury_Description"))
    patient = _first(accident.get("Patient"))
    model_record = build_model_record(patient=patient, accident=accident)
    for key, output in run_models(model_record, INTAKE_MODELS).items():
        accident[RESPONSE_KEYS[key]] = {**output, "prediction_id": None, "persisted": False}
    return accident


def predict_and_store(
    supabase: Client,
    *,
    accident: dict[str, Any],
    patient: dict[str, Any],
    model_keys: tuple[str, ...],
    trigger_event: str,
    requested_by: Any,
    transfers: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run the given models for one accident and append them to prediction history."""
    model_record = build_model_record(patient=patient, accident=accident, transfers=transfers)
    results = run_models(model_record, model_keys)
    prediction_store.store_predictions(
        supabase,
        accident_id=accident.get("accident_id"),
        hospital_id=accident.get("hospital_id"),
        record=model_record,
        predictions=results,
        trigger_event=trigger_event,
        requested_by=requested_by,
    )
    return results


def add_patient_predictions_many(accidents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [add_patient_predictions(accident) for accident in accidents]


# Backwards-compatible names used by the existing routers.
add_severity_prediction = add_patient_predictions
add_severity_predictions = add_patient_predictions_many


def build_discharge_record(
    *,
    patient: dict[str, Any],
    accident: dict[str, Any],
    transfers: list[dict[str, Any]],
) -> dict[str, Any]:
    return build_model_record(patient=patient, accident=accident, transfers=transfers)
