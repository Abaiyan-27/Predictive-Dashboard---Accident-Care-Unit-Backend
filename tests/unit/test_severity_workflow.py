"""Severity workflow rules that need no database: inputs, states, locks, audit."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.contracts.severity import SEVERITY_LABELS
from app.schemas.accident import SeverityDecisionIn, UpdateIntakeForm
from app.services import audit_service, ml_service, patient_service, severity_workflow as wf

ACCIDENT = {
    "accident_id": "acc-1",
    "hospital_id": "hosp-a",
    "incident_at": "2026-09-01T19:00:00+00:00",
    "collision_with": "motorbike",
    "mode_of_travel_during_accident": "three wheeler passenger",
    "vehicle_type": "motor cyclist",
    "helmet_worn": "no",
    "passenger_type": "driver",
    "mode_of_transport": "other vehicle",
    "time_to_hospital": "30 minutes - 1 hour",
    "other_injury": "no other injury",
    "road_type": "bend",
    "Patient": {"full_name": "A", "age": "25", "gender": "Female"},
    "Injury_Description": [{"injury_order": 1, "site_of_injury": "chest injury", "type_of_injury": "contusion"}],
    "Treatment": [],
}


def prediction(label, status, *, accident=ACCIDENT, seq=1, hospital="hosp-a", inputs_hash=None):
    return {
        "prediction_id": f"p{seq}", "prediction_seq": seq, "hospital_id": hospital, "model_key": "severity",
        "status": status, "prediction": label,
        "inputs_hash": inputs_hash or wf.current_inputs_hash(wf.model_record(accident)),
    }


def decision(prediction_id, final, source, seq=1):
    return {"decision_id": f"d{seq}", "decision_seq": seq, "prediction_id": prediction_id,
            "final_decision": final, "decision_source": source, "decided_by": "nurse-1"}


# ── Inputs ─────────────────────────────────────────────────────────────────

def test_severity_fields_are_exactly_the_models_raw_inputs():
    assert set(wf.SEVERITY_INPUT_FIELDS.values()) == set(ml_service.model_input_fields("severity"))
    assert not set(wf.SEVERITY_INPUT_FIELDS) & set(wf.LATER_MODEL_FIELDS)


@pytest.mark.parametrize("count", range(0, 5))
def test_readiness_matches_when_the_model_refuses(count):
    raw_inputs = ml_service.model_input_fields("severity")
    values = patient_service.build_model_record(patient=ACCIDENT["Patient"], accident=ACCIDENT)
    record = {name: (values[name] if index < count else None) for index, name in enumerate(raw_inputs)}
    readiness = ml_service.severity_readiness(record)
    assert readiness["available"] is True
    assert len(readiness["present"]) == count
    refused = ml_service.predict_severity(record=record).status == "REFUSED"
    assert readiness["ready"] is (not refused)


def test_input_summary_names_missing_fields_by_form_name():
    summary = wf.input_summary(wf.model_record(ACCIDENT))
    assert summary["recorded"] == 11 and summary["total"] == 12
    assert summary["missing"] == ["secondary_site_of_injury"]
    assert summary["ready"] is True and summary["complete"] is False


# ── States ─────────────────────────────────────────────────────────────────

def test_states_without_a_prediction():
    assert wf.episode_status(ACCIDENT, [], {}, role="nurse")["state"] == wf.NOT_RUN
    bare = {**ACCIDENT, **{field: None for field in wf.SEVERITY_INPUT_FIELDS},
            "Patient": {"full_name": "A"}, "Injury_Description": []}
    status = wf.episode_status(bare, [], {}, role="nurse")
    assert status["state"] == wf.WAITING_FOR_INPUTS
    assert status["can_run"] is False and status["basic_complete"] is False


@pytest.mark.parametrize("label, status, state, can_accept", [
    ("HIGH", "OK", wf.PREDICTION_AVAILABLE, True),
    ("BORDERLINE", "LOW_CONFIDENCE", wf.PREDICTION_AVAILABLE, True),
    ("LOW", "OK", wf.PREDICTION_AVAILABLE, True),
    ("DEFER TO CLINICAL JUDGEMENT", "REFUSED", wf.MODEL_DEFERRED, False),
    ("DEFER TO CLINICAL JUDGEMENT", "ERROR", wf.PREDICTION_FAILED, False),
    ("DEFER TO CLINICAL JUDGEMENT", "UNAVAILABLE", wf.PREDICTION_FAILED, False),
])
def test_prediction_states(label, status, state, can_accept):
    result = wf.episode_status(ACCIDENT, [prediction(label, status)], {}, role="nurse")
    assert result["state"] == state
    assert result["prediction"]["prediction"] == label
    assert result["prediction"]["is_defer"] is (label == "DEFER TO CLINICAL JUDGEMENT")
    assert (result["can_accept"], result["can_override"]) == (can_accept, True)
    assert result["final_severity"] is None


def test_low_and_defer_are_distinct_states():
    low = wf.episode_status(ACCIDENT, [prediction("LOW", "OK")], {}, role="nurse")
    defer = wf.episode_status(ACCIDENT, [prediction("DEFER TO CLINICAL JUDGEMENT", "REFUSED")], {}, role="nurse")
    assert low["state"] != defer["state"]
    assert low["prediction"]["prediction"] != defer["prediction"]["prediction"]


@pytest.mark.parametrize("source, final, state", [("MODEL", "HIGH", wf.ACCEPTED), ("NURSE", "LOW", wf.OVERRIDDEN)])
def test_decided_states(source, final, state):
    result = wf.episode_status(ACCIDENT, [prediction("HIGH", "OK")], {"p1": [decision("p1", final, source)]},
                               role="nurse")
    assert result["state"] == state and result["final_severity"] == final
    assert result["decision"]["model_prediction"] == "HIGH"
    assert result["locks"]["severity_inputs"] is True
    assert (result["can_run"], result["can_accept"], result["can_override"]) == (False, False, False)


def test_changed_inputs_make_the_prediction_stale():
    result = wf.episode_status(ACCIDENT, [prediction("HIGH", "OK", inputs_hash="0" * 64)], {}, role="nurse")
    assert result["state"] == wf.NOT_RUN
    assert result["prediction"]["stale"] is True and result["can_accept"] is False


def test_prediction_from_another_hospital_cannot_be_decided_here():
    result = wf.episode_status(ACCIDENT, [prediction("HIGH", "OK", hospital="hosp-b")], {}, role="nurse")
    assert result["prediction"]["stale"] is True and result["can_accept"] is False


@pytest.mark.parametrize("role", ["doctor", "admin", None])
def test_only_nurses_get_workflow_actions(role):
    result = wf.episode_status(ACCIDENT, [prediction("HIGH", "OK")], {}, role=role)
    assert (result["can_run"], result["can_accept"], result["can_override"]) == (False, False, False)


def test_states_never_use_retired_labels():
    states = {wf.NOT_RUN, wf.WAITING_FOR_INPUTS, wf.PREDICTION_AVAILABLE, wf.MODEL_DEFERRED,
              wf.PREDICTION_FAILED, wf.ACCEPTED, wf.OVERRIDDEN}
    assert len(states) == 7
    for text in (*states, *SEVERITY_LABELS):
        for retired in ("PRIORITY", "MODERATE", "MEDIUM", "NOT CLASSIFIED", "NOT_CLASSIFIED"):
            assert retired not in text.upper()


# ── Locks ──────────────────────────────────────────────────────────────────

def locked_status(*, severity=True, later=False):
    accident = {**ACCIDENT, "Treatment": [{"treatment_id": "t"}] if later else []}
    decisions = {"p1": [decision("p1", "HIGH", "MODEL")]} if severity else {}
    return accident, wf.episode_status(accident, [prediction("HIGH", "OK", accident=accident)], decisions, role="nurse")


def test_changing_a_locked_severity_input_is_a_violation():
    accident, status = locked_status()
    assert wf.lock_violations(UpdateIntakeForm(age="60"), accident, status) == ["age"]
    assert wf.lock_violations(UpdateIntakeForm(incident_at="2026-09-01T08:00:00Z"), accident, status) == ["incident_at"]


def test_resending_stored_values_and_basic_details_is_allowed():
    accident, status = locked_status()
    form = UpdateIntakeForm(full_name="New", gender="Male", age="25", incident_at="2026-09-01T19:00:00Z",
                            collision_with="motorbike", road_type="junction",
                            injuries=[{"injury_order": 1, "site_of_injury": "chest injury", "type_of_injury": "fracture"}])
    assert wf.lock_violations(form, accident, status) == []


def test_injury_list_that_drops_a_locked_site_is_a_violation():
    accident, status = locked_status()
    form = UpdateIntakeForm(injuries=[{"injury_order": 2, "site_of_injury": "knee"}])
    assert set(wf.lock_violations(form, accident, status)) == {"site_of_injury", "secondary_site_of_injury"}


def test_doctor_model_inputs_lock_after_treatment_only():
    accident, status = locked_status(severity=False, later=False)
    assert wf.lock_violations(UpdateIntakeForm(road_type="junction"), accident, status) == []
    accident, status = locked_status(severity=False, later=True)
    assert wf.lock_violations(UpdateIntakeForm(road_type="junction", age="60"), accident, status) == ["road_type"]


# ── Decision payload ───────────────────────────────────────────────────────

@pytest.mark.parametrize("payload", [
    {"action": "override", "final_decision": "DEFER TO CLINICAL JUDGEMENT"},
    {"action": "override", "final_decision": "Moderate"},
    {"action": "override"},
    {"action": "accept", "final_decision": "HIGH"},
])
def test_decision_payload_validation(payload):
    with pytest.raises(ValueError):
        SeverityDecisionIn(**payload)


def test_human_decision_labels_are_the_actionable_contract_labels():
    for label in ("HIGH", "BORDERLINE", "LOW"):
        assert SeverityDecisionIn(action="override", final_decision=label).final_decision == label


# ── Audit ──────────────────────────────────────────────────────────────────

def test_audit_event_never_raises_and_falls_back_without_details():
    supabase = MagicMock()
    calls = []

    def insert(entry):
        calls.append(entry)
        if "details" in entry:
            raise RuntimeError("column details does not exist")
        return MagicMock(execute=MagicMock(return_value=MagicMock(data=[entry])))

    supabase.table.return_value.insert.side_effect = insert
    written = audit_service.log_event(supabase, actor_id="u", role="nurse", action_type="x", details={"a": 1})
    assert written == {"actor_id": "u", "role": "nurse", "action_type": "x"}
    assert len(calls) == 2

    supabase.table.return_value.insert.side_effect = RuntimeError("down")
    assert audit_service.log_event(supabase, actor_id="u", role="nurse", action_type="x") == {}
