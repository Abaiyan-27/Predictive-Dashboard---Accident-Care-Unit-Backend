"""The nurse workflow and Severity decisions against the real API and a migrated database.

Everything goes through the HTTP API with the real Severity model; the
database is inspected directly to prove what was stored. Hospital A
(DGH – Vavuniya) has two nurses, a doctor and an admin; hospital B
(DGH – Mannar) has one nurse; there is also a government officer.
"""
from __future__ import annotations

import json
import uuid

import pytest

from app.core.security import create_access_token
from app.services import ml_service

from .conftest import psycopg, running_api

HOSPITAL_A, HOSPITAL_B = "DGH – Vavuniya", "DGH – Mannar"

# Inputs the deployed Severity model scores HIGH (p ≈ 0.38) and BORDERLINE (p ≈ 0.29).
HIGH_INPUTS = {
    "age": "25",
    "incident_at": "2026-09-01T19:00:00+00:00",
    "collision_with": "motorbike",
    "mode_of_travel_during_accident": "three wheeler passenger",
    "vehicle_type": "motor cyclist",
    "helmet_worn": "no",
    "passenger_type": "driver",
    "mode_of_transport": "other vehicle",
    "time_to_hospital": "30 minutes - 1 hour",
    "other_injury": "no other injury",
    "injuries": [{"injury_order": 1, "site_of_injury": "chest injury", "type_of_injury": "contusion"}],
}
BORDERLINE_CHANGES = {
    "collision_with": "train",
    "helmet_worn": "yes",
    "passenger_type": "rearseatpassenger",
    "mode_of_transport": "ambulance",
    "injuries": [{"injury_order": 1, "site_of_injury": "hand", "type_of_injury": "contusion"}],
}
TREATMENT = {
    "number_of_days_stay": 2, "investigation_done": "x-ray",
    "management_name_1": "wound suturing", "management_name_2": "no other management",
}
FORBIDDEN_LABELS = ("Priority", "Moderate", "Medium", "Severity not classified")


@pytest.fixture(scope="module")
def world(pg_cluster):
    db = pg_cluster.create_database("nurse_severity")
    pg_cluster.migrate(db)
    # Supabase sessions run in UTC and the time-of-collision bucket depends on
    # it; set it before PostgREST opens its connection pool.
    with psycopg.connect(pg_cluster.dsn(db), autocommit=True) as setup:
        setup.execute(f'alter database "{db}" set timezone to \'UTC\'')
    with running_api(pg_cluster, db) as (client, conn):
        hospitals = {
            name: str(conn.execute('select hospital_id from "Hospital" where name = %s', (name,)).fetchone()[0])
            for name in (HOSPITAL_A, HOSPITAL_B)
        }
        people: dict[str, dict] = {}
        for key, role, hospital in [
            ("nurse_a1", "nurse", HOSPITAL_A), ("nurse_a2", "nurse", HOSPITAL_A),
            ("doctor_a", "doctor", HOSPITAL_A), ("admin_a", "admin", HOSPITAL_A),
            ("nurse_b", "nurse", HOSPITAL_B),
        ]:
            user_id = str(uuid.uuid4())
            conn.execute('insert into "User" (user_id, name, email, role, hospital_id) values (%s, %s, %s, %s, %s)',
                         (user_id, key, f"{key}@example.org", role, hospitals[hospital]))
            people[key] = {"id": user_id, "role": role, "hospital": hospitals[hospital]}
        officer = str(uuid.uuid4())
        conn.execute('insert into "Government_Officer" (officer_id, name, email) values (%s, %s, %s)',
                     (officer, "Officer", "officer@example.org"))
        people["government"] = {"id": officer, "role": "government", "hospital": None}
        for person in people.values():
            token = create_access_token({"sub": person["id"], "role": person["role"]})
            person["headers"] = {"Authorization": f"Bearer {token}"}
        yield {"client": client, "db": conn, "people": people, "hospitals": hospitals, "responses": []}


# ── Helpers ────────────────────────────────────────────────────────────────

def call(world, who, method, path, **kwargs):
    response = world["client"].request(method, path, headers=world["people"][who]["headers"], **kwargs)
    if response.status_code < 400:  # a 422 echoes the invalid label a test sent on purpose
        world["responses"].append(response.text)
    return response


def register(world, nurse="nurse_a1", **fields):
    response = call(world, nurse, "POST", "/api/patients/intake",
                    json={"full_name": f"Patient {uuid.uuid4().hex[:6]}", "gender": "Female", **fields})
    assert response.status_code == 201, response.text
    return response.json()["accident_record"]["accident_id"]


def run(world, accident, who="nurse_a1"):
    return call(world, who, "POST", f"/api/severity/accidents/{accident}/run")


def decide(world, prediction_id, who="nurse_a1", **body):
    return call(world, who, "POST", f"/api/severity/predictions/{prediction_id}/decision", json=body)


def predicted(world, accident, who="nurse_a1"):
    response = run(world, accident, who)
    assert response.status_code == 200, response.text
    return response.json()


def stored_predictions(world, accident, model_key=None):
    query = ('select prediction_id::text, model_key, status, prediction, probability, inputs, '
             'requested_by::text, hospital_id::text from "Model_Prediction" where accident_id = %s')
    params = [accident]
    if model_key:
        query += " and model_key = %s"
        params.append(model_key)
    return world["db"].execute(query + " order by prediction_seq", params).fetchall()


def stored_decisions(world, accident):
    return world["db"].execute(
        'select d.prediction_id::text, d.final_decision, d.decision_source, d.decided_by::text, d.decided_at '
        'from "Prediction_Decision" d join "Model_Prediction" p using (prediction_id) '
        "where p.accident_id = %s order by d.decision_seq", (accident,)).fetchall()


def audit(world, accident, action):
    return world["db"].execute(
        'select actor_id::text, role, hospital_id::text, details, "timestamp" from "AuditLog" '
        "where accident_id = %s and action_type = %s order by \"timestamp\"", (accident, action)).fetchall()


class FixedProbability:
    """Wraps the deployed estimator but returns a chosen escalation probability.

    Used only where the deployed model cannot reach an output on any input
    (see test_low_output_*), so the real thresholding code is still exercised.
    """

    def __init__(self, model, positive):
        self.classes_ = model.classes_
        self.positive = positive

    def predict_proba(self, rows):
        import numpy as np
        classes = [str(value) for value in self.classes_]
        row = [0.0] * len(classes)
        positive_index = classes.index("1")
        row[positive_index] = self.positive
        row[1 - positive_index] = 1 - self.positive
        return np.array([row] * len(rows))


def fix_probability(monkeypatch, positive):
    real = ml_service._load_bundle_model

    def load(bundle_name, contract):
        loaded, error = real(bundle_name, contract)
        if bundle_name == "severity" and isinstance(loaded, dict):
            loaded = {**loaded, "model": FixedProbability(loaded["model"], positive)}
        return loaded, error

    monkeypatch.setattr(ml_service, "_load_bundle_model", load)


# ── Partial intake: save, return, complete ────────────────────────────────

def test_nurse_creates_patient_with_basic_details_only(world):
    response = call(world, "nurse_a1", "POST", "/api/patients/intake",
                    json={"full_name": "Basic Only", "gender": "Male"})
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["accident_record"]["hospital_id"] == world["hospitals"][HOSPITAL_A]
    severity = body["severity"]
    assert severity["state"] == "WAITING_FOR_INPUTS"
    assert severity["basic_complete"] is False          # age not given yet
    assert severity["inputs"]["recorded"] == 0 and severity["inputs"]["ready"] is False
    assert stored_predictions(world, body["accident_record"]["accident_id"]) == []


def test_partial_intake_can_be_completed_over_several_sessions(world):
    accident = register(world)
    # Session 2: the same nurse adds the age.
    response = call(world, "nurse_a1", "PATCH", f"/api/patients/intake/{accident}", json={"age": "25"})
    assert response.status_code == 200, response.text
    assert response.json()["severity"]["basic_complete"] is True
    assert response.json()["severity"]["state"] == "WAITING_FOR_INPUTS"   # 1 of 12 inputs
    # Session 3: another nurse of the same hospital adds some Severity inputs.
    response = call(world, "nurse_a2", "PATCH", f"/api/patients/intake/{accident}",
                    json={"collision_with": "motorbike", "time_to_hospital": "30 minutes - 1 hour"})
    assert response.status_code == 200, response.text
    severity = response.json()["severity"]
    assert severity["state"] == "NOT_RUN"
    assert severity["inputs"]["recorded"] == 3 and severity["inputs"]["ready"] is True
    assert severity["inputs"]["complete"] is False
    # Session 4: the rest, still without running the model.
    rest = {key: value for key, value in HIGH_INPUTS.items() if key not in {"age", "collision_with", "time_to_hospital"}}
    severity = call(world, "nurse_a1", "PATCH", f"/api/patients/intake/{accident}", json=rest).json()["severity"]
    assert severity["inputs"]["recorded"] == 11 and severity["state"] == "NOT_RUN"
    assert severity["inputs"]["missing"] == ["secondary_site_of_injury"]
    assert stored_predictions(world, accident) == []   # saving never runs a model


def test_nurse_sees_and_updates_another_nurses_patient(world):
    accident = register(world, "nurse_a1", **HIGH_INPUTS)
    listed = {row["accident_id"]: row for row in call(world, "nurse_a2", "GET", "/api/patients/my").json()}
    assert accident in listed
    assert listed[accident]["severity_workflow"]["state"] == "NOT_RUN"
    response = call(world, "nurse_a2", "PATCH", f"/api/patients/intake/{accident}",
                    json={"full_name": "Renamed By Nurse A2", "road_type": "bend"})
    assert response.status_code == 200, response.text
    assert response.json()["patient"]["full_name"] == "Renamed By Nurse A2"
    # A nurse of another hospital neither sees nor edits it.
    assert accident not in {row["accident_id"] for row in call(world, "nurse_b", "GET", "/api/patients/my").json()}
    assert call(world, "nurse_b", "PATCH", f"/api/patients/intake/{accident}",
                json={"full_name": "x"}).status_code == 404


# ── Severity runs only when enough inputs exist ───────────────────────────

def test_severity_waits_for_required_inputs(world):
    accident = register(world, age="25")
    response = run(world, accident)
    assert response.status_code == 422
    assert "Missing:" in response.json()["detail"]
    assert stored_predictions(world, accident) == []
    status = call(world, "nurse_a1", "GET", f"/api/severity/accidents/{accident}").json()
    assert status["state"] == "WAITING_FOR_INPUTS" and status["can_run"] is False


def test_real_severity_model_runs_and_prediction_is_stored(world):
    accident = register(world, **HIGH_INPUTS)
    body = predicted(world, accident)
    assert body["state"] == "PREDICTION_AVAILABLE"
    prediction = body["prediction"]
    assert prediction["prediction"] == "HIGH" and prediction["status"] == "OK"
    rows = stored_predictions(world, accident)
    assert len(rows) == 1
    prediction_id, model_key, status, label, probability, inputs, requested_by, hospital = rows[0]
    assert (prediction_id, model_key, label) == (prediction["prediction_id"], "severity", "HIGH")
    assert requested_by == world["people"]["nurse_a1"]["id"]
    assert hospital == world["hospitals"][HOSPITAL_A]
    # The stored output is what the deployed model returns for the stored inputs.
    rerun = ml_service.predict_severity(record=inputs)
    assert (rerun.prediction, rerun.status) == (label, status)
    assert float(probability) == pytest.approx(rerun.probability, abs=1e-4)
    executed = audit(world, accident, "severity_prediction_executed")
    assert len(executed) == 1
    assert executed[0][3]["prediction_id"] == prediction_id and executed[0][3]["model_prediction"] == "HIGH"


def test_nurse_never_runs_doctor_models(world):
    accident = register(world, **HIGH_INPUTS)
    predicted(world, accident)
    call(world, "nurse_a1", "PATCH", f"/api/patients/intake/{accident}", json={"road_type": "junction"})
    assert {row[1] for row in stored_predictions(world, accident)} == {"severity"}
    patient = world["db"].execute('select patient_id::text from "Accident_Record" where accident_id = %s',
                                  (accident,)).fetchone()[0]
    assert call(world, "nurse_a1", "GET", f"/api/patients/{patient}/prognosis").status_code == 403
    assert call(world, "nurse_a1", "POST", "/api/xai/explain",
                json={"model_name": "stay_duration", "accident_id": accident}).status_code == 403
    listed = next(row for row in call(world, "nurse_a1", "GET", "/api/patients/my").json()
                  if row["accident_id"] == accident)
    assert not {"stay_duration_prediction", "transfer_prediction", "discharge_prediction"} & set(listed)


@pytest.mark.parametrize("inputs, expected, status", [
    (HIGH_INPUTS, "HIGH", "OK"),
    ({**HIGH_INPUTS, **BORDERLINE_CHANGES}, "BORDERLINE", "LOW_CONFIDENCE"),
])
def test_real_model_outputs_can_be_accepted(world, inputs, expected, status):
    accident = register(world, **inputs)
    prediction = predicted(world, accident)["prediction"]
    assert (prediction["prediction"], prediction["status"]) == (expected, status)
    response = decide(world, prediction["prediction_id"], action="accept")
    assert response.status_code == 201, response.text
    assert response.json()["decision"]["final_decision"] == expected


def test_low_output_is_not_reached_by_the_deployed_model_on_its_training_data():
    """Documents the deployed model: no training-data input scores below the LOW cut.

    A full scan of the 5,555 distinct input rows in the notebook dataset gave
    only HIGH (2,707) and BORDERLINE (2,848); the lowest escalation probability
    was 0.1146 against low_cut 0.08. This samples the first 150 rows.
    """
    import pandas as pd
    from pathlib import Path

    dataset = Path(__file__).resolve().parents[3] / "Group-01---Notebooks" / "Dataset.csv"
    if not dataset.exists():
        pytest.skip("notebook dataset not available")
    contract, _ = ml_service._read_contract("severity")
    bundle, _ = ml_service._load_bundle_model("severity", contract)
    frame = pd.read_csv(dataset)[bundle["raw_inputs"]].drop_duplicates().head(150)
    labels = {ml_service.predict_severity(record=row.where(row.notna(), None).to_dict()).prediction
              for _, row in frame.iterrows()}
    assert "LOW" not in labels
    assert labels <= {"HIGH", "BORDERLINE"}


def test_low_output_is_stored_and_accepted_as_low(world, monkeypatch):
    # The real thresholding code maps a probability below low_cut to LOW.
    fix_probability(monkeypatch, 0.03)
    accident = register(world, **HIGH_INPUTS)
    prediction = predicted(world, accident)["prediction"]
    assert (prediction["prediction"], prediction["status"], prediction["is_defer"]) == ("LOW", "OK", False)
    response = decide(world, prediction["prediction_id"], action="accept")
    assert response.status_code == 201, response.text
    assert response.json()["state"] == "ACCEPTED"
    assert stored_decisions(world, accident)[0][1:3] == ("LOW", "MODEL")


# ── Accept and override ───────────────────────────────────────────────────

def test_accept_records_model_prediction_as_final_decision(world):
    accident = register(world, **HIGH_INPUTS)
    prediction_id = predicted(world, accident)["prediction"]["prediction_id"]
    response = decide(world, prediction_id, action="accept")
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["state"] == "ACCEPTED" and body["final_severity"] == "HIGH"
    assert body["decision"]["decision_source"] == "MODEL"
    assert body["decision"]["prediction_id"] == prediction_id

    [(decided_on, final, source, decided_by, decided_at)] = stored_decisions(world, accident)
    assert (decided_on, final, source) == (prediction_id, "HIGH", "MODEL")
    assert decided_by == world["people"]["nurse_a1"]["id"] and decided_at is not None
    assert stored_predictions(world, accident)[0][3] == "HIGH"

    [entry] = audit(world, accident, "severity_prediction_accepted")
    actor, role, hospital, details, timestamp = entry
    assert (actor, role, hospital) == (world["people"]["nurse_a1"]["id"], "nurse", world["hospitals"][HOSPITAL_A])
    assert details["prediction_id"] == prediction_id and details["final_decision"] == "HIGH"
    assert timestamp is not None


def test_override_keeps_model_prediction_and_records_nurse_decision(world):
    accident = register(world, **HIGH_INPUTS)
    prediction_id = predicted(world, accident)["prediction"]["prediction_id"]
    response = decide(world, prediction_id, action="override", final_decision="LOW", reason="Stable on review")
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["state"] == "OVERRIDDEN" and body["final_severity"] == "LOW"
    assert body["decision"]["model_prediction"] == "HIGH"
    assert body["prediction"]["prediction"] == "HIGH"

    [(decided_on, final, source, decided_by, _)] = stored_decisions(world, accident)
    assert (decided_on, final, source, decided_by) == (prediction_id, "LOW", "NURSE", world["people"]["nurse_a1"]["id"])
    assert stored_predictions(world, accident)[0][3] == "HIGH"   # original prediction unchanged

    [entry] = audit(world, accident, "severity_prediction_overridden")
    actor, role, hospital, details, timestamp = entry
    assert actor == world["people"]["nurse_a1"]["id"] and hospital == world["hospitals"][HOSPITAL_A]
    assert details["prediction_id"] == prediction_id
    assert (details["model_prediction"], details["final_decision"]) == ("HIGH", "LOW")
    assert details["reason"] == "Stable on review" and timestamp is not None


def test_override_with_the_model_label_must_be_an_accept(world):
    accident = register(world, **HIGH_INPUTS)
    prediction_id = predicted(world, accident)["prediction"]["prediction_id"]
    assert decide(world, prediction_id, action="override", final_decision="HIGH").status_code == 422
    assert stored_decisions(world, accident) == []


@pytest.mark.parametrize("body", [
    {"action": "override", "final_decision": "DEFER TO CLINICAL JUDGEMENT"},
    {"action": "override", "final_decision": "Moderate"},
    {"action": "override", "final_decision": "Priority"},
    {"action": "override", "final_decision": "MEDIUM"},
    {"action": "override", "final_decision": "low"},
    {"action": "override"},
    {"action": "accept", "final_decision": "LOW"},
    {"action": "approve"},
])
def test_invalid_decisions_are_rejected(world, body):
    accident = register(world, **HIGH_INPUTS)
    prediction_id = predicted(world, accident)["prediction"]["prediction_id"]
    assert decide(world, prediction_id, **body).status_code == 422
    assert stored_decisions(world, accident) == []


def test_identity_and_hospital_in_the_request_body_are_ignored(world):
    accident = register(world, **HIGH_INPUTS)
    prediction_id = predicted(world, accident)["prediction"]["prediction_id"]
    response = decide(world, prediction_id, action="override", final_decision="BORDERLINE",
                      decided_by=world["people"]["doctor_a"]["id"], decision_source="MODEL",
                      role="doctor", hospital_id=world["hospitals"][HOSPITAL_B])
    assert response.status_code == 201, response.text
    [(_, final, source, decided_by, _)] = stored_decisions(world, accident)
    assert (final, source, decided_by) == ("BORDERLINE", "NURSE", world["people"]["nurse_a1"]["id"])


# ── DEFER states ──────────────────────────────────────────────────────────

def test_failed_prediction_defers_and_nurse_makes_clinical_decision(world, monkeypatch):
    accident = register(world, **HIGH_INPUTS)

    def broken(_frame):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(ml_service, "_engineer_severity_row", broken)
    body = predicted(world, accident)
    assert body["state"] == "PREDICTION_FAILED"
    assert body["prediction"]["prediction"] == "DEFER TO CLINICAL JUDGEMENT"
    assert body["prediction"]["status"] == "ERROR"
    assert (body["can_accept"], body["can_override"]) == (False, True)
    prediction_id = body["prediction"]["prediction_id"]

    assert decide(world, prediction_id, action="accept").status_code == 422
    response = decide(world, prediction_id, action="override", final_decision="HIGH")
    assert response.status_code == 201, response.text
    assert response.json()["state"] == "OVERRIDDEN" and response.json()["final_severity"] == "HIGH"
    assert stored_predictions(world, accident)[0][3] == "DEFER TO CLINICAL JUDGEMENT"


def test_model_refusal_is_its_own_state_and_low_stays_distinct_from_defer(world, monkeypatch):
    # A REFUSED prediction (the model declining to score, as stored by earlier
    # intake runs) is shown as MODEL_DEFERRED, not as a failure or as LOW.
    # Simulate the bundle refusing this record by raising its completeness floor
    # above 100% for one run; the backend's own gate is bypassed for that run.
    accident = register(world, **HIGH_INPUTS)
    real_load = ml_service._load_bundle_model

    def refusing_bundle(bundle_name, contract):
        loaded, error = real_load(bundle_name, contract)
        if bundle_name == "severity" and isinstance(loaded, dict):
            loaded = {**loaded, "completeness_floor": 1.01}
        return loaded, error

    monkeypatch.setattr(ml_service, "_load_bundle_model", refusing_bundle)
    monkeypatch.setattr(ml_service, "severity_readiness", lambda record: {
        "available": True, "ready": True, "raw_inputs": [], "present": [], "missing": [],
        "minimum": 0, "message": None})
    body = predicted(world, accident)
    monkeypatch.undo()
    assert body["prediction"]["status"] == "REFUSED"
    assert body["state"] == "MODEL_DEFERRED"
    assert body["prediction"]["prediction"] == "DEFER TO CLINICAL JUDGEMENT"
    prediction_id = body["prediction"]["prediction_id"]

    assert decide(world, prediction_id, action="accept").status_code == 422
    response = decide(world, prediction_id, action="override", final_decision="LOW")
    assert response.status_code == 201, response.text
    final = response.json()
    assert (final["final_severity"], final["prediction"]["prediction"]) == ("LOW", "DEFER TO CLINICAL JUDGEMENT")
    assert stored_decisions(world, accident)[0][1:3] == ("LOW", "NURSE")


# ── Who may decide ────────────────────────────────────────────────────────

@pytest.fixture
def open_prediction(world):
    accident = register(world, **HIGH_INPUTS)
    return accident, predicted(world, accident)["prediction"]["prediction_id"]


@pytest.mark.parametrize("who", ["doctor_a", "admin_a", "government"])
def test_only_nurses_decide_severity(world, open_prediction, who):
    accident, prediction_id = open_prediction
    assert decide(world, prediction_id, who=who, action="accept").status_code == 403
    assert decide(world, prediction_id, who=who, action="override", final_decision="LOW").status_code == 403
    assert stored_decisions(world, accident) == []


@pytest.mark.parametrize("who", ["doctor_a", "admin_a", "government"])
def test_only_nurses_run_severity(world, open_prediction, who):
    accident, _ = open_prediction
    assert run(world, accident, who).status_code == 403
    assert len(stored_predictions(world, accident)) == 1


def test_nurse_from_another_hospital_cannot_decide_or_run(world, open_prediction):
    accident, prediction_id = open_prediction
    assert decide(world, prediction_id, who="nurse_b", action="accept").status_code == 404
    assert run(world, accident, "nurse_b").status_code == 404
    assert call(world, "nurse_b", "GET", f"/api/severity/accidents/{accident}").status_code == 404
    assert stored_decisions(world, accident) == []


def test_decision_on_a_nonexistent_prediction_is_not_found(world):
    assert decide(world, str(uuid.uuid4()), action="accept").status_code == 404
    assert decide(world, "not-a-uuid", action="accept").status_code == 422


def test_decision_on_a_non_severity_prediction_is_rejected(world):
    accident = register(world, **HIGH_INPUTS)
    response = call(world, "doctor_a", "POST", "/api/treatments", json={
        **TREATMENT, "accident_id": accident, "hospital_id": world["hospitals"][HOSPITAL_A]})
    assert response.status_code == 201, response.text
    stay_id = stored_predictions(world, accident, "stay_duration")[-1][0]
    assert decide(world, stay_id, action="accept").status_code == 404
    assert decide(world, stay_id, action="override", final_decision="LOW").status_code == 404
    assert stored_decisions(world, accident) == []


def test_other_staff_can_view_but_not_act_on_severity(world, open_prediction):
    accident, _ = open_prediction
    for who in ("doctor_a", "admin_a"):
        status = call(world, who, "GET", f"/api/severity/accidents/{accident}")
        assert status.status_code == 200
        body = status.json()
        assert body["prediction"]["prediction"] == "HIGH"
        assert (body["can_run"], body["can_accept"], body["can_override"]) == (False, False, False)
    assert call(world, "government", "GET", f"/api/severity/accidents/{accident}").status_code == 403


# ── History and stale screens ─────────────────────────────────────────────

def test_repeated_runs_keep_history_and_decisions_target_the_exact_prediction(world):
    accident = register(world, **HIGH_INPUTS)
    first = predicted(world, accident)["prediction"]
    assert first["prediction"] == "HIGH"

    # Re-running unchanged inputs reuses the stored prediction.
    assert predicted(world, accident)["prediction"]["prediction_id"] == first["prediction_id"]

    # Changing an input makes the shown prediction stale until Severity runs again.
    changed = call(world, "nurse_a1", "PATCH", f"/api/patients/intake/{accident}", json=BORDERLINE_CHANGES).json()
    assert changed["severity"]["prediction"]["stale"] is True
    assert changed["severity"]["state"] == "NOT_RUN"
    stale = decide(world, first["prediction_id"], action="accept")
    assert stale.status_code == 409 and "inputs changed" in stale.json()["detail"]

    second = predicted(world, accident)["prediction"]
    assert second["prediction_id"] != first["prediction_id"] and second["prediction"] == "BORDERLINE"

    history = stored_predictions(world, accident)
    assert [row[0] for row in history] == [first["prediction_id"], second["prediction_id"]]
    assert history[0][3] == "HIGH"        # the old prediction is unchanged

    # A screen still showing the first prediction cannot decide on it.
    old_screen = decide(world, first["prediction_id"], action="override", final_decision="LOW")
    assert old_screen.status_code == 409 and "newer Severity prediction" in old_screen.json()["detail"]

    response = decide(world, second["prediction_id"], action="accept")
    assert response.status_code == 201, response.text
    [(decided_on, final, source, _, _)] = stored_decisions(world, accident)
    assert (decided_on, final, source) == (second["prediction_id"], "BORDERLINE", "MODEL")

    listed = call(world, "nurse_a1", "GET", f"/api/accidents/{accident}/predictions")
    assert listed.status_code == 200
    assert [row["prediction_id"] for row in listed.json()] == [first["prediction_id"], second["prediction_id"]]
    assert listed.json()[1]["decisions"][0]["final_decision"] == "BORDERLINE"


def test_database_rejects_an_update_to_a_decided_prediction(world, open_prediction):
    accident, prediction_id = open_prediction
    assert decide(world, prediction_id, action="override", final_decision="LOW").status_code == 201
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with world["db"].transaction():
            world["db"].execute('update "Model_Prediction" set prediction = %s where prediction_id = %s',
                                ("LOW", prediction_id))
    assert stored_predictions(world, accident)[0][3] == "HIGH"


# ── Editing rules after the decision ──────────────────────────────────────

def test_decided_severity_inputs_are_locked_but_basic_details_stay_editable(world, open_prediction):
    accident, prediction_id = open_prediction
    assert decide(world, prediction_id, action="accept").status_code == 201
    path = f"/api/patients/intake/{accident}"

    for change in ({"age": "60"}, {"collision_with": "train"}, {"incident_at": "2026-09-01T08:00:00+00:00"},
                   {"injuries": [{"injury_order": 1, "site_of_injury": "hand", "type_of_injury": "contusion"}]},
                   {"injuries": [{"injury_order": 2, "site_of_injury": "knee"}]}):
        response = call(world, "nurse_a1", "PATCH", path, json=change)
        assert response.status_code == 409, (change, response.text)
        assert "cannot be changed" in response.json()["detail"]

    row = world["db"].execute(
        'select p.age, a.collision_with, i.site_of_injury from "Accident_Record" a join "Patient" p using (patient_id) '
        'join "Injury_Description" i on i.accident_id = a.accident_id and i.injury_order = 1 '
        "where a.accident_id = %s", (accident,)).fetchone()
    assert row == ("25", "motorbike", "chest injury")

    # Re-sending stored values, basic details and doctor-model inputs are allowed.
    same = call(world, "nurse_a1", "PATCH", path, json={**HIGH_INPUTS, "full_name": "Edited Name", "gender": "Male"})
    assert same.status_code == 200, same.text
    assert call(world, "nurse_a1", "PATCH", path, json={"road_type": "junction", "injuries": [
        {"injury_order": 1, "site_of_injury": "chest injury", "type_of_injury": "fracture"}]}).status_code == 200

    status = call(world, "nurse_a1", "GET", f"/api/severity/accidents/{accident}").json()
    assert status["state"] == "ACCEPTED" and status["locks"]["severity_inputs"] is True
    assert "age" in status["locks"]["fields"] and "full_name" not in status["locks"]["fields"]
    # The decision is final: no new run and no second decision.
    assert run(world, accident).status_code == 409
    assert decide(world, prediction_id, action="override", final_decision="LOW").status_code == 409
    assert len(stored_decisions(world, accident)) == 1


def test_doctor_model_inputs_lock_once_treatment_is_recorded(world):
    accident = register(world, **HIGH_INPUTS, road_type="bend")
    response = call(world, "doctor_a", "POST", "/api/treatments", json={
        **TREATMENT, "accident_id": accident, "hospital_id": world["hospitals"][HOSPITAL_A]})
    assert response.status_code == 201, response.text
    path = f"/api/patients/intake/{accident}"
    blocked = call(world, "nurse_a1", "PATCH", path, json={"road_type": "junction"})
    assert blocked.status_code == 409 and "road_type" in blocked.json()["detail"]
    assert call(world, "nurse_a1", "PATCH", path, json={"full_name": "Still Editable"}).status_code == 200
    # Severity inputs stay editable until the nurse decides.
    assert call(world, "nurse_a1", "PATCH", path, json={"helmet_worn": "yes"}).status_code == 200


# ── Audit failures do not block clinical work ─────────────────────────────

def test_audit_failure_does_not_fail_the_decision(world, open_prediction, monkeypatch):
    accident, prediction_id = open_prediction
    from app.services import audit_service

    def unavailable(*args, **kwargs):
        raise RuntimeError("audit store down")

    monkeypatch.setattr(audit_service, "log_action", unavailable)
    response = decide(world, prediction_id, action="accept")
    assert response.status_code == 201, response.text
    assert len(stored_decisions(world, accident)) == 1


# ── Labels ────────────────────────────────────────────────────────────────

def test_no_response_contains_a_retired_severity_label(world):
    # Runs last in this module: every response the nurse workflow returned above.
    assert world["responses"]
    for text in world["responses"]:
        payload = json.dumps(json.loads(text)) if text.startswith(("{", "[")) else text
        for label in FORBIDDEN_LABELS:
            assert label not in payload, (label, payload[:300])
