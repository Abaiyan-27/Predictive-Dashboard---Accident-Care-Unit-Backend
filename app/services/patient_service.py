"""Helpers for enriching Supabase accident records with real model outputs."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from app.services import ml_service


def _first(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        return value[0] if value else {}
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return [value] if isinstance(value, dict) else []


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


def build_model_record(
    *,
    patient: dict[str, Any],
    accident: dict[str, Any],
    transfers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    injuries = _list(accident.get("Injury_Description"))
    treatments = _list(accident.get("Treatment"))
    known_transfers = transfers if transfers is not None else _list(accident.get("Transfer"))
    first_injury = injuries[0] if injuries else {}
    second_injury = injuries[1] if len(injuries) > 1 else {}
    latest_treatment = max(
        treatments,
        key=lambda treatment: str(treatment.get("created_at") or ""),
    ) if treatments else {}
    latest_transfer = max(
        known_transfers,
        key=lambda transfer: str(transfer.get("requested_at") or transfer.get("created_at") or ""),
    ) if known_transfers else {}
    total_days = sum(
        int(treatment.get("number_of_days_stay") or 0)
        for treatment in treatments
    ) if treatments else None

    return {
        # Arrival-time severity-triage inputs.
        "Time of Collision": _collision_time_bucket(accident.get("incident_at")),
        "Age": patient.get("age"),
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
        # Post-treatment stay, transfer, and discharge inputs.
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
        "Transfer To Next Hospital": "yes" if known_transfers else "no",
        "State of Transfer": latest_transfer.get("state_of_transfer"),
        "Investigation Done": latest_treatment.get("investigation_done"),
        "Management Name 1": latest_treatment.get("management_name_1"),
        "Management Name 2": latest_treatment.get("management_name_2"),
        "total_days": total_days,
    }


def add_patient_predictions(accident: dict[str, Any]) -> dict[str, Any]:
    patient = _first(accident.get("Patient"))
    model_record = build_model_record(patient=patient, accident=accident)

    severity = ml_service.predict_severity(record=model_record)
    stay_duration = ml_service.predict_stay_duration(model_record)
    transfer = ml_service.predict_transfer_probability(model_record)

    accident["severity_prediction"] = severity.to_dict()
    accident["stay_duration_prediction"] = stay_duration.to_dict()
    accident["transfer_prediction"] = transfer.to_dict()
    return accident


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
