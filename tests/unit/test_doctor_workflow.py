"""Doctor workflow rules that need no database: readiness, states, staleness."""
from __future__ import annotations

import pytest

from app.schemas.accident import ModelDecisionIn
from app.services import doctor_workflow as dw
from app.services import ml_service

ACCIDENT = {
    "accident_id": "acc-1", "hospital_id": "hosp-a",
    "incident_at": "2026-09-01T19:00:00+00:00", "mode_of_travel_during_accident": "motor cyclist",
    "collision_with": "heavy vehicle", "vehicle_type": "motor cyclist", "helmet_worn": "no",
    "passenger_type": "driver", "mode_of_transport": "ambulance", "time_to_hospital": "30 minutes - 1 hour",
    "road_type": "bend", "speed": "40 - 80 km/h", "road_condition": "good", "visibility": "adequate",
    "road_signals_exist": "no", "collision_force_from": "front", "alcohol_consumption": "no", "illicit_drugs": "no",
    "Patient": {"age": "8"},
    "Injury_Description": [{"injury_order": 1, "site_of_injury": "chest injury", "type_of_injury": "dislocation"}],
    "Treatment": [{"number_of_days_stay": 1, "investigation_done": "mri scan", "created_at": "2026-10-03",
                   "management_name_1": "only medicine and discharge", "management_name_2": "strict bed rest"}],
    "Transfer": [],
}
UNTREATED = {**ACCIDENT, "Treatment": []}


def stored(model_key, label, status, *, accident=ACCIDENT, hospital="hosp-a", inputs_hash=None, seq=1):
    return {
        "prediction_id": f"{model_key}-{seq}", "prediction_seq": seq, "hospital_id": hospital, "model_key": model_key,
        "status": status, "prediction": label,
        "inputs_hash": inputs_hash or dw.current_inputs_hash(model_key, dw.model_record(accident)),
    }


def status(model_key, predictions, decisions=None, *, accident=ACCIDENT, role="doctor"):
    return dw.model_status(model_key, accident, dw.model_record(accident), predictions, decisions or {}, role=role)


@pytest.mark.parametrize("model_key", dw.DOCTOR_MODELS)
@pytest.mark.parametrize("accident", [ACCIDENT, UNTREATED, {**ACCIDENT, "road_type": None, "speed": None,
                                                            "road_condition": None, "visibility": None}])
def test_readiness_agrees_with_the_models_own_refusal(model_key, accident):
    record = dw.model_record(accident)
    refused = ml_service.predict_bundle(model_key, record).status == "REFUSED"
    assert ml_service.bundle_readiness(model_key, record)["ready"] is (not refused)


def test_labels_match_model_outputs():
    assert set(dw.DECISION_LABELS) == set(dw.DOCTOR_MODELS)
    record = dw.model_record(ACCIDENT)
    for model_key in dw.DOCTOR_MODELS:
        assert ml_service.predict_bundle(model_key, record).prediction in dw.DECISION_LABELS[model_key]


@pytest.mark.parametrize("model_key", dw.DOCTOR_MODELS)
def test_waiting_until_treatment(model_key):
    result = status(model_key, [], accident=UNTREATED)
    assert result["state"] == dw.WAITING_FOR_INPUTS
    assert "Investigation Done" in result["readiness"]["missing_required"]
    assert result["can_run"] is False


@pytest.mark.parametrize("label_status, state, can_accept", [
    ("OK", dw.PREDICTION_AVAILABLE, True),
    ("LOW_CONFIDENCE", dw.PREDICTION_AVAILABLE, True),
    ("ERROR", dw.PREDICTION_FAILED, False),
])
def test_prediction_states(label_status, state, can_accept):
    label = "2-3 days" if label_status != "ERROR" else None
    result = status("stay_duration", [stored("stay_duration", label, label_status)])
    assert result["state"] == state
    assert (result["can_accept"], result["can_override"]) == (can_accept, True)


def test_a_refused_row_is_not_a_prediction():
    result = status("transfer", [stored("transfer", None, "REFUSED")])
    assert result["state"] == dw.NOT_RUN and result["prediction"] is None


def test_changed_inputs_and_other_hospital_make_a_prediction_stale():
    for prediction in (stored("discharge", "Partial recovery", "OK", inputs_hash="0" * 64),
                       stored("discharge", "Partial recovery", "OK", hospital="hosp-b")):
        result = status("discharge", [prediction])
        assert result["prediction"]["stale"] is True
        assert result["state"] == dw.NOT_RUN and result["can_accept"] is False


@pytest.mark.parametrize("source, state", [("MODEL", dw.ACCEPTED), ("DOCTOR", dw.OVERRIDDEN)])
def test_decided_states_are_final(source, state):
    prediction = stored("transfer", "Transfer", "OK")
    final = "Transfer" if source == "MODEL" else "No transfer"
    decision = {"decision_id": "d", "decision_seq": 1, "prediction_id": prediction["prediction_id"],
                "final_decision": final, "decision_source": source}
    result = status("transfer", [prediction], {prediction["prediction_id"]: [decision]})
    assert result["state"] == state and result["final_decision"] == final
    assert result["decision"]["model_prediction"] == "Transfer"
    assert (result["can_run"], result["can_accept"], result["can_override"]) == (False, False, False)
    models = {key: status(key, []) for key in dw.DOCTOR_MODELS}
    models["transfer"] = result
    assert dw.treatment_status(ACCIDENT, models, role="doctor")["locked"] is True


@pytest.mark.parametrize("role", ["nurse", "admin", None])
def test_only_doctors_get_actions(role):
    result = status("stay_duration", [stored("stay_duration", "2-3 days", "OK")], role=role)
    assert (result["can_run"], result["can_accept"], result["can_override"]) == (False, False, False)


def test_override_needs_a_decision():
    with pytest.raises(ValueError):
        ModelDecisionIn(action="override")
    assert ModelDecisionIn(action="accept").final_decision is None
