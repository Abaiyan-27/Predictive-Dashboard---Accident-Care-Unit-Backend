"""The doctor workflow and the three doctor-model decisions, against the real API,
the real model artifacts and a migrated database.

Hospital A (DGH – Vavuniya): nurse, two doctors, admin. Hospital B (DGH – Mannar):
nurse and doctor. Plus a government officer. Patients are admitted by the
nurse through the API, treated by the doctor, then decided model by model.
"""
from __future__ import annotations

import json
import uuid

import pytest

from app.core.security import create_access_token
from app.services import doctor_workflow, ml_service

from .conftest import psycopg, running_api

HOSPITAL_A, HOSPITAL_B = "DGH – Vavuniya", "DGH – Mannar"
DOCTOR_MODELS = ("stay_duration", "transfer", "discharge")

CONTEXT = {
    "incident_at": "2026-09-01T19:00:00+00:00", "mode_of_travel_during_accident": "motor cyclist",
    "collision_with": "heavy vehicle", "vehicle_type": "motor cyclist", "helmet_worn": "no",
    "passenger_type": "driver", "mode_of_transport": "ambulance", "time_to_hospital": "30 minutes - 1 hour",
    "other_injury": "no other injury", "road_type": "bend", "road_condition": "good", "visibility": "adequate",
    "road_signals_exist": "no", "collision_force_from": "front", "alcohol_consumption": "no", "illicit_drugs": "no",
}
# Real model outputs for these inputs (found by scoring the deployed artifacts):
#   CASE_A: Stay 2-3 days, Transfer No transfer, Discharge Partial recovery
#   CASE_B: Stay 4+ days,  Transfer Transfer,    Discharge Complete recovery
CASE_A = {
    "intake": {**CONTEXT, "age": "8", "speed": "40 - 80 km/h",
               "injuries": [{"injury_order": 1, "site_of_injury": "chest injury", "type_of_injury": "dislocation"}]},
    "treatment": {"number_of_days_stay": 1, "investigation_done": "mri scan",
                  "management_name_1": "only medicine and discharge", "management_name_2": "strict bed rest"},
}
CASE_B = {
    "intake": {**CONTEXT, "age": "70", "speed": "less than 40 km/h",
               "injuries": [{"injury_order": 1, "site_of_injury": "forearm", "type_of_injury": "abration"}]},
    "treatment": {"number_of_days_stay": 2, "investigation_done": "others",
                  "management_name_1": "fracture stabilization", "management_name_2": "pop applied"},
}
RETIRED = ("Priority", "Moderate", "Medium", "Severity not classified")


@pytest.fixture(scope="module")
def world(pg_cluster):
    db = pg_cluster.create_database("doctor_workflow")
    pg_cluster.migrate(db)
    # Supabase runs in UTC; set it before PostgREST opens its pool.
    with psycopg.connect(pg_cluster.dsn(db), autocommit=True) as setup:
        setup.execute(f'alter database "{db}" set timezone to \'UTC\'')
    with running_api(pg_cluster, db) as (client, conn):
        hospitals = {
            name: str(conn.execute('select hospital_id from "Hospital" where name = %s', (name,)).fetchone()[0])
            for name in (HOSPITAL_A, HOSPITAL_B)
        }
        people: dict[str, dict] = {}
        for key, role, hospital in [
            ("nurse_a", "nurse", HOSPITAL_A), ("doctor_a", "doctor", HOSPITAL_A),
            ("doctor_a2", "doctor", HOSPITAL_A), ("admin_a", "admin", HOSPITAL_A),
            ("nurse_b", "nurse", HOSPITAL_B), ("doctor_b", "doctor", HOSPITAL_B),
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
            person["headers"] = {"Authorization": f"Bearer {create_access_token({'sub': person['id'], 'role': person['role']})}"}
        yield {"client": client, "db": conn, "people": people, "hospitals": hospitals, "responses": []}


# ── Helpers ────────────────────────────────────────────────────────────────

def call(world, who, method, path, **kwargs):
    response = world["client"].request(method, path, headers=world["people"][who]["headers"], **kwargs)
    if response.status_code < 400:
        world["responses"].append(response.text)
    return response


def admit(world, case, nurse="nurse_a", severity_decision=None):
    """Nurse intake + Severity run (+ optional nurse decision). Returns the accident id."""
    body = call(world, nurse, "POST", "/api/patients/intake",
                json={"full_name": f"Patient {uuid.uuid4().hex[:6]}", "gender": "Male", **case["intake"]})
    assert body.status_code == 201, body.text
    accident = body.json()["accident_record"]["accident_id"]
    run = call(world, nurse, "POST", f"/api/severity/accidents/{accident}/run")
    assert run.status_code == 200, run.text
    if severity_decision:
        prediction_id = run.json()["prediction"]["prediction_id"]
        decided = call(world, nurse, "POST", f"/api/severity/predictions/{prediction_id}/decision", json=severity_decision)
        assert decided.status_code == 201, decided.text
    return accident


def treat(world, accident, case, nurse="nurse_a", **extra):
    """Treatment is a model input, so a nurse records it (doctors only review and decide)."""
    return call(world, nurse, "POST", "/api/treatments", json={"accident_id": accident, **case["treatment"], **extra})


def treated(world, case, **kwargs):
    accident = admit(world, case, **kwargs)
    response = treat(world, accident, case)
    assert response.status_code == 201, response.text
    return accident


def view(world, accident, who="doctor_a"):
    response = call(world, who, "GET", f"/api/doctor/accidents/{accident}")
    assert response.status_code == 200, response.text
    return response.json()


def prediction_of(world, accident, model_key):
    return view(world, accident)["models"][model_key]["prediction"]


def decide(world, accident, prediction_id, who="doctor_a", **body):
    return call(world, who, "POST", f"/api/doctor/accidents/{accident}/predictions/{prediction_id}/decision", json=body)


def stored(world, accident, model_key=None):
    query = ('select prediction_id::text, model_key, status, prediction, inputs, trigger_event, requested_by::text, '
             'hospital_id::text from "Model_Prediction" where accident_id = %s')
    params = [accident]
    if model_key:
        query += " and model_key = %s"
        params.append(model_key)
    return world["db"].execute(query + " order by prediction_seq", params).fetchall()


def decisions(world, accident, model_key=None):
    query = ('select d.prediction_id::text, d.model_key, d.final_decision, d.decision_source, d.decided_by::text, '
             'd.decided_at, d.reason from "Prediction_Decision" d join "Model_Prediction" p using (prediction_id) '
             "where p.accident_id = %s")
    params = [accident]
    if model_key:
        query += " and d.model_key = %s"
        params.append(model_key)
    return world["db"].execute(query + " order by d.decision_seq", params).fetchall()


def audit(world, accident, action):
    return world["db"].execute(
        'select actor_id::text, role, hospital_id::text, details, "timestamp" from "AuditLog" '
        'where accident_id = %s and action_type = %s order by "timestamp"', (accident, action)).fetchall()


# ── Labels ─────────────────────────────────────────────────────────────────

def test_decision_labels_match_the_database_and_model_contracts(world):
    for model_key, labels in doctor_workflow.DECISION_LABELS.items():
        rows = world["db"].execute(
            'select label from "Model_Output_Label" where model_key = %s and human_selectable order by sort_order',
            (model_key,)).fetchall()
        assert tuple(row[0] for row in rows) == labels
    for model_key in ("transfer", "discharge"):
        contract, _ = ml_service._read_contract(model_key)
        assert set(contract["label_map"].values()) == set(doctor_workflow.DECISION_LABELS[model_key])


# ── Severity is visible but read-only ─────────────────────────────────────

def test_doctor_sees_severity_and_nurse_decision_but_cannot_change_it(world):
    accident = admit(world, CASE_A)
    severity_id = view(world, accident)["severity"]["prediction"]["prediction_id"]
    model_label = view(world, accident)["severity"]["prediction"]["prediction"]
    nurse_choice = next(label for label in ("HIGH", "BORDERLINE", "LOW") if label != model_label)
    response = call(world, "nurse_a", "POST", f"/api/severity/predictions/{severity_id}/decision",
                    json={"action": "override", "final_decision": nurse_choice})
    assert response.status_code == 201, response.text

    severity = view(world, accident)["severity"]
    assert severity["prediction"]["prediction"] == model_label
    assert severity["decision"]["final_decision"] == nurse_choice
    assert severity["decision"]["decision_source"] == "NURSE"
    assert (severity["can_run"], severity["can_accept"], severity["can_override"]) == (False, False, False)

    before = decisions(world, accident)
    # Through the nurse endpoint, the doctor endpoint, a re-run and an intake edit: all refused.
    assert call(world, "doctor_a", "POST", f"/api/severity/predictions/{severity_id}/decision",
                json={"action": "accept"}).status_code == 403
    assert decide(world, accident, severity_id, action="override", final_decision="HIGH").status_code == 403
    assert call(world, "doctor_a", "POST", f"/api/severity/accidents/{accident}/run").status_code == 403
    assert call(world, "doctor_a", "PATCH", f"/api/patients/intake/{accident}", json={"age": "60"}).status_code == 403
    assert decisions(world, accident) == before


# ── Readiness and treatment ───────────────────────────────────────────────

def test_doctor_models_wait_for_treatment(world):
    accident = admit(world, CASE_A)
    episode = view(world, accident)
    for model_key in DOCTOR_MODELS:
        status = episode["models"][model_key]
        assert status["state"] == "WAITING_FOR_INPUTS"
        assert "Investigation Done" in status["readiness"]["missing_required"]
        assert status["prediction"] is None
    assert episode["treatment"]["recorded"] is False and episode["treatment"]["can_record"] is False   # nurses record it
    assert call(world, "doctor_a", "POST", f"/api/doctor/accidents/{accident}/run").status_code == 422
    # The prognosis view stores nothing for models that are not ready, and never runs Severity.
    patient = world["db"].execute('select patient_id::text from "Accident_Record" where accident_id = %s',
                                  (accident,)).fetchone()[0]
    assert call(world, "doctor_a", "GET", f"/api/patients/{patient}/prognosis").status_code == 200
    assert {row[1] for row in stored(world, accident)} == {"severity"}


def test_treatment_runs_the_real_doctor_models_and_stores_the_predictions(world):
    accident = admit(world, CASE_A)
    severity_rows = stored(world, accident, "severity")
    response = treat(world, accident, CASE_A)       # no hospital_id sent
    assert response.status_code == 201, response.text
    treatment = world["db"].execute(
        'select hospital_id::text, logged_by::text, decided_by::text from "Treatment" where accident_id = %s',
        (accident,)).fetchone()
    assert treatment == (world["hospitals"][HOSPITAL_A], world["people"]["nurse_a"]["id"], None)

    rows = {row[1]: row for row in stored(world, accident) if row[1] in DOCTOR_MODELS}
    assert set(rows) == set(DOCTOR_MODELS)
    assert stored(world, accident, "severity") == severity_rows   # Severity is not the doctor's model
    expected = {"stay_duration": "2-3 days", "transfer": "No transfer", "discharge": "Partial recovery"}
    episode = view(world, accident)
    for model_key, row in rows.items():
        prediction_id, _, status, label, inputs, trigger, requested_by, hospital = row
        assert label == expected[model_key]
        assert (trigger, requested_by, hospital) == (
            "treatment_recorded", world["people"]["nurse_a"]["id"], world["hospitals"][HOSPITAL_A])
        assert inputs["Investigation Done"] == "mri scan" and inputs["Management Name 1"] == "only medicine and discharge"
        rerun = ml_service.predict_bundle(model_key, inputs)          # the artifact, on the stored inputs
        assert (rerun.prediction, rerun.status) == (label, status)
        shown = episode["models"][model_key]
        assert shown["state"] == "PREDICTION_AVAILABLE"
        assert shown["prediction"]["prediction_id"] == prediction_id and shown["prediction"]["prediction"] == label

    assert audit(world, accident, "treatment_logged")[0][3]["investigation_done"] == "mri scan"
    for model_key in DOCTOR_MODELS:
        [entry] = audit(world, accident, f"{model_key}_prediction_executed")
        assert entry[3]["prediction_id"] == rows[model_key][0] and entry[0] == world["people"]["nurse_a"]["id"]


@pytest.mark.parametrize("who", ["doctor_a", "doctor_a2", "admin_a", "government"])
def test_only_nurses_enter_treatment(world, who):
    """Every model input comes from a nurse: doctors (and others) cannot record treatment."""
    accident = admit(world, CASE_A)
    assert call(world, who, "POST", "/api/treatments", json={"accident_id": accident, **CASE_A["treatment"]}).status_code == 403
    assert call(world, who, "GET", f"/api/accidents/{accident}/treatment").status_code == 403
    assert world["db"].execute('select count(*) from "Treatment" where accident_id = %s', (accident,)).fetchone()[0] == 0


def test_nurse_sees_the_treatment_status_and_its_lock(world):
    accident = admit(world, CASE_A)
    before = call(world, "nurse_a", "GET", f"/api/accidents/{accident}/treatment").json()
    assert (before["recorded"], before["locked"], before["can_record"]) == (False, False, True)
    assert treat(world, accident, CASE_A).status_code == 201
    after = call(world, "nurse_a", "GET", f"/api/accidents/{accident}/treatment").json()
    assert after["recorded"] is True and after["records"][0]["investigation_done"] == "mri scan"
    stay = prediction_of(world, accident, "stay_duration")
    assert decide(world, accident, stay["prediction_id"], action="accept").status_code == 201
    locked = call(world, "nurse_a", "GET", f"/api/accidents/{accident}/treatment").json()
    assert (locked["locked"], locked["can_record"]) == (True, False)
    assert call(world, "nurse_b", "GET", f"/api/accidents/{accident}/treatment").status_code == 404


def test_treatment_can_name_the_deciding_doctor_of_the_same_hospital(world):
    accident = admit(world, CASE_A)
    people = world["people"]
    for decider in ("nurse_a", "doctor_b"):
        assert treat(world, accident, CASE_A, decided_by=people[decider]["id"]).status_code == 403
    assert treat(world, accident, CASE_A, decided_by=people["doctor_a2"]["id"]).status_code == 201
    stored_decider = world["db"].execute('select decided_by::text from "Treatment" where accident_id = %s',
                                         (accident,)).fetchone()[0]
    assert stored_decider == people["doctor_a2"]["id"]


def test_treatment_hospital_comes_from_the_account(world):
    accident = admit(world, CASE_A)
    response = treat(world, accident, CASE_A, hospital_id=world["hospitals"][HOSPITAL_B])
    assert response.status_code == 403
    assert world["db"].execute('select count(*) from "Treatment" where accident_id = %s', (accident,)).fetchone()[0] == 0


def test_repeated_runs_and_new_treatment_keep_history(world):
    accident = treated(world, CASE_A)
    first = {row[1]: row[0] for row in stored(world, accident) if row[1] in DOCTOR_MODELS}
    # Running again on unchanged inputs reuses the stored predictions.
    assert call(world, "doctor_a", "POST", f"/api/doctor/accidents/{accident}/run").status_code == 200
    assert {row[1]: row[0] for row in stored(world, accident) if row[1] in DOCTOR_MODELS} == first
    # A second treatment (before any decision) changes the inputs: new rows, old rows intact.
    response = treat(world, accident, CASE_B)
    assert response.status_code == 201, response.text
    history = [row for row in stored(world, accident) if row[1] in DOCTOR_MODELS]
    assert len(history) == 6
    assert {row[0] for row in history[:3]} == set(first.values())
    assert all(row[5] == "treatment_recorded" for row in history)
    current = view(world, accident)["models"]
    assert {current[key]["prediction"]["prediction_id"] for key in DOCTOR_MODELS} == {row[0] for row in history[3:]}


# ── Accept / override, model by model ─────────────────────────────────────

@pytest.mark.parametrize("model_key, case, model_label", [
    ("stay_duration", CASE_A, "2-3 days"),
    ("transfer", CASE_B, "Transfer"),
    ("discharge", CASE_A, "Partial recovery"),
])
def test_accept_records_the_model_prediction(world, model_key, case, model_label):
    accident = treated(world, case)
    prediction = prediction_of(world, accident, model_key)
    assert prediction["prediction"] == model_label
    response = decide(world, accident, prediction["prediction_id"], action="accept")
    assert response.status_code == 201, response.text
    status = response.json()["models"][model_key]
    assert status["state"] == "ACCEPTED" and status["final_decision"] == model_label

    [(decided_on, key, final, source, decided_by, decided_at, _)] = decisions(world, accident, model_key)
    assert (decided_on, key, final, source) == (prediction["prediction_id"], model_key, model_label, "MODEL")
    assert decided_by == world["people"]["doctor_a"]["id"] and decided_at is not None
    assert stored(world, accident, model_key)[-1][3] == model_label            # prediction unchanged

    [entry] = audit(world, accident, f"{model_key}_prediction_accepted")
    actor, role, hospital, details, timestamp = entry
    assert (actor, role, hospital) == (world["people"]["doctor_a"]["id"], "doctor", world["hospitals"][HOSPITAL_A])
    assert details["model"] == model_key and details["prediction_id"] == prediction["prediction_id"]
    assert (details["model_prediction"], details["final_decision"], details["decision_source"]) == (
        model_label, model_label, "MODEL")
    assert timestamp is not None


@pytest.mark.parametrize("model_key, case, model_label, doctor_label", [
    ("stay_duration", CASE_A, "2-3 days", "4+ days"),
    ("transfer", CASE_B, "Transfer", "No transfer"),
    ("discharge", CASE_A, "Partial recovery", "Complete recovery"),
])
def test_override_keeps_the_prediction_and_records_the_doctor_decision(world, model_key, case, model_label, doctor_label):
    accident = treated(world, case)
    prediction = prediction_of(world, accident, model_key)
    assert prediction["prediction"] == model_label
    response = decide(world, accident, prediction["prediction_id"], action="override",
                      final_decision=doctor_label, reason="Clinical review")
    assert response.status_code == 201, response.text
    status = response.json()["models"][model_key]
    assert status["state"] == "OVERRIDDEN" and status["final_decision"] == doctor_label
    assert status["decision"]["model_prediction"] == model_label

    [(decided_on, _, final, source, decided_by, _, reason)] = decisions(world, accident, model_key)
    assert (decided_on, final, source, reason) == (prediction["prediction_id"], doctor_label, "DOCTOR", "Clinical review")
    assert decided_by == world["people"]["doctor_a"]["id"]
    assert stored(world, accident, model_key)[-1][3] == model_label            # original prediction remains

    [entry] = audit(world, accident, f"{model_key}_prediction_overridden")
    details = entry[3]
    assert (details["model_prediction"], details["final_decision"], details["decision_source"]) == (
        model_label, doctor_label, "DOCTOR")
    assert details["reason"] == "Clinical review"


def test_the_three_decisions_are_independent(world):
    accident = treated(world, CASE_B)
    episode = view(world, accident)
    ids = {key: episode["models"][key]["prediction"]["prediction_id"] for key in DOCTOR_MODELS}
    labels = {key: episode["models"][key]["prediction"]["prediction"] for key in DOCTOR_MODELS}
    assert decide(world, accident, ids["stay_duration"], action="accept").status_code == 201
    after_stay = view(world, accident)["models"]
    assert after_stay["stay_duration"]["state"] == "ACCEPTED"
    assert after_stay["transfer"]["state"] == after_stay["discharge"]["state"] == "PREDICTION_AVAILABLE"

    other = next(label for label in ("Transfer", "No transfer") if label != labels["transfer"])
    assert decide(world, accident, ids["transfer"], action="override", final_decision=other).status_code == 201
    assert decide(world, accident, ids["discharge"], action="accept").status_code == 201

    rows = decisions(world, accident)
    assert [(row[0], row[1], row[3]) for row in rows] == [
        (ids["stay_duration"], "stay_duration", "MODEL"),
        (ids["transfer"], "transfer", "DOCTOR"),
        (ids["discharge"], "discharge", "MODEL"),
    ]
    final = view(world, accident)["models"]
    assert [final[key]["state"] for key in DOCTOR_MODELS] == ["ACCEPTED", "OVERRIDDEN", "ACCEPTED"]


@pytest.mark.parametrize("model_key, body", [
    ("stay_duration", {"action": "override", "final_decision": "5 days"}),
    ("stay_duration", {"action": "override", "final_decision": "4+ Days"}),
    ("stay_duration", {"action": "override", "final_decision": "Transfer"}),
    ("stay_duration", {"action": "override", "final_decision": "HIGH"}),
    ("transfer", {"action": "override", "final_decision": "Maybe"}),
    ("transfer", {"action": "override", "final_decision": "4+ days"}),
    ("transfer", {"action": "override", "final_decision": "Priority"}),
    ("discharge", {"action": "override", "final_decision": "Recovered"}),
    ("discharge", {"action": "override", "final_decision": "Moderate"}),
    ("discharge", {"action": "override", "final_decision": "No transfer"}),
    ("discharge", {"action": "override"}),
    ("transfer", {"action": "approve"}),
])
def test_invalid_decisions_are_rejected(world, model_key, body):
    accident = treated(world, CASE_A)
    prediction_id = prediction_of(world, accident, model_key)["prediction_id"]
    assert decide(world, accident, prediction_id, **body).status_code == 422
    assert decisions(world, accident, model_key) == []


@pytest.mark.parametrize("model_key", DOCTOR_MODELS)
def test_accept_must_match_and_override_must_differ(world, model_key):
    accident = treated(world, CASE_A)
    prediction = prediction_of(world, accident, model_key)
    other = next(label for label in doctor_workflow.DECISION_LABELS[model_key] if label != prediction["prediction"])
    accept_other = decide(world, accident, prediction["prediction_id"], action="accept", final_decision=other)
    assert accept_other.status_code == 422
    same = decide(world, accident, prediction["prediction_id"], action="override",
                  final_decision=prediction["prediction"])
    assert same.status_code == 422
    assert decisions(world, accident, model_key) == []
    # Accepting while naming the model's own label is fine.
    response = decide(world, accident, prediction["prediction_id"], action="accept",
                      final_decision=prediction["prediction"])
    assert response.status_code == 201, response.text


def test_identity_in_the_request_body_is_ignored(world):
    accident = treated(world, CASE_A)
    prediction_id = prediction_of(world, accident, "stay_duration")["prediction_id"]
    response = decide(world, accident, prediction_id, action="override", final_decision="0-1 days",
                      decided_by=world["people"]["doctor_b"]["id"], doctor_id=world["people"]["doctor_b"]["id"],
                      hospital_id=world["hospitals"][HOSPITAL_B], role="admin", decision_source="MODEL")
    assert response.status_code == 201, response.text
    [(_, _, final, source, decided_by, _, _)] = decisions(world, accident, "stay_duration")
    assert (final, source, decided_by) == ("0-1 days", "DOCTOR", world["people"]["doctor_a"]["id"])


# ── Stale screens and final decisions ─────────────────────────────────────

def test_stale_predictions_are_refused_until_the_model_runs_again(world):
    # Severity not decided, so the nurse may still edit its inputs; Transfer reads
    # "Time Taken To Reach Hospital", so its prediction goes stale.
    accident = treated(world, CASE_A)
    old = prediction_of(world, accident, "transfer")
    edit = call(world, "nurse_a", "PATCH", f"/api/patients/intake/{accident}",
                json={"time_to_hospital": "more than 2 hour"})
    assert edit.status_code == 200, edit.text

    status = view(world, accident)["models"]["transfer"]
    assert status["prediction"]["stale"] is True and status["state"] == "NOT_RUN"
    response = decide(world, accident, old["prediction_id"], action="accept")
    assert response.status_code == 409 and "inputs changed" in response.json()["detail"]

    assert call(world, "doctor_a", "POST", f"/api/doctor/accidents/{accident}/run").status_code == 200
    new = prediction_of(world, accident, "transfer")
    assert new["prediction_id"] != old["prediction_id"] and new["stale"] is False
    older_screen = decide(world, accident, old["prediction_id"], action="accept")
    assert older_screen.status_code == 409 and "newer" in older_screen.json()["detail"]
    assert stored(world, accident, "transfer")[0][0] == old["prediction_id"]   # history kept

    assert decide(world, accident, new["prediction_id"], action="accept").status_code == 201
    assert decisions(world, accident, "transfer")[0][0] == new["prediction_id"]


def test_a_decision_is_final(world):
    accident = treated(world, CASE_A)
    prediction_id = prediction_of(world, accident, "discharge")["prediction_id"]
    assert decide(world, accident, prediction_id, action="accept").status_code == 201
    again = decide(world, accident, prediction_id, who="doctor_a2", action="override", final_decision="Complete recovery")
    assert again.status_code == 409 and "already been decided" in again.json()["detail"]
    assert len(decisions(world, accident, "discharge")) == 1
    # New treatment would change a decided prediction's inputs, so it is locked.
    episode = view(world, accident)
    assert episode["treatment"]["locked"] is True and episode["treatment"]["can_record"] is False
    assert treat(world, accident, CASE_B).status_code == 409
    assert world["db"].execute('select count(*) from "Treatment" where accident_id = %s', (accident,)).fetchone()[0] == 1
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with world["db"].transaction():
            world["db"].execute('update "Prediction_Decision" set final_decision = %s where prediction_id = %s',
                                ("Complete recovery", prediction_id))


def test_prediction_from_another_episode_is_not_found(world):
    first = treated(world, CASE_A)
    second = treated(world, CASE_A)
    foreign = prediction_of(world, second, "stay_duration")["prediction_id"]
    assert decide(world, first, foreign, action="accept").status_code == 404
    assert decide(world, first, str(uuid.uuid4()), action="accept").status_code == 404
    assert decisions(world, second) == []


# ── Authorization ─────────────────────────────────────────────────────────

@pytest.fixture
def episode_a(world):
    accident = treated(world, CASE_A)
    return accident, {key: prediction_of(world, accident, key)["prediction_id"] for key in DOCTOR_MODELS}


def test_same_hospital_doctors_can_work_on_the_episode(world, episode_a):
    accident, ids = episode_a
    for who in ("doctor_a", "doctor_a2"):
        assert call(world, who, "GET", f"/api/doctor/accidents/{accident}").status_code == 200
        assert call(world, who, "GET", f"/api/predictions/{ids['transfer']}").status_code == 200
    listed = call(world, "doctor_a2", "GET", f"/api/patients/hospital/{world['hospitals'][HOSPITAL_A]}").json()
    row = next(item for item in listed if item["accident_id"] == accident)
    assert row["doctor_workflow"]["models"]["stay_duration"]["prediction"] == "2-3 days"
    assert "severity_prediction" not in row and "stay_duration_prediction" not in row
    assert decide(world, accident, ids["transfer"], who="doctor_a2", action="accept").status_code == 201


def test_other_hospital_doctor_is_denied_everything(world, episode_a):
    accident, ids = episode_a
    assert call(world, "doctor_b", "GET", f"/api/doctor/accidents/{accident}").status_code == 404
    assert call(world, "doctor_b", "GET", f"/api/predictions/{ids['stay_duration']}").status_code == 404
    assert call(world, "doctor_b", "POST", f"/api/doctor/accidents/{accident}/run").status_code == 404
    for model_key in DOCTOR_MODELS:
        assert decide(world, accident, ids[model_key], who="doctor_b", action="accept").status_code == 404
    assert treat(world, accident, CASE_B, nurse="nurse_b").status_code == 404
    assert treat(world, accident, CASE_B, nurse="nurse_b",
                 hospital_id=world["hospitals"][HOSPITAL_A]).status_code == 403
    assert decisions(world, accident) == []


@pytest.mark.parametrize("who", ["nurse_a", "admin_a", "government", "nurse_b"])
def test_only_doctors_decide_or_run_doctor_models(world, episode_a, who):
    accident, ids = episode_a
    for model_key in DOCTOR_MODELS:
        label = doctor_workflow.DECISION_LABELS[model_key][0]
        assert decide(world, accident, ids[model_key], who=who, action="accept").status_code == 403
        assert decide(world, accident, ids[model_key], who=who, action="override",
                      final_decision=label).status_code == 403
    assert call(world, who, "POST", f"/api/doctor/accidents/{accident}/run").status_code == 403
    assert call(world, who, "GET", f"/api/doctor/accidents/{accident}").status_code == 403
    assert decisions(world, accident) == []


def test_database_refuses_a_doctor_decision_from_a_nurse(world, episode_a):
    accident, ids = episode_a
    with pytest.raises(psycopg.errors.CheckViolation):
        with world["db"].transaction():
            world["db"].execute(
                'insert into "Prediction_Decision" (prediction_id, model_key, final_decision, decision_source, '
                "decided_by) values (%s, 'stay_duration', '4+ days', 'NURSE', %s)",
                (ids["stay_duration"], world["people"]["nurse_a"]["id"]))


# ── Labels in responses ───────────────────────────────────────────────────

def test_no_response_contains_a_retired_label(world):
    assert world["responses"]
    for text in world["responses"]:
        payload = json.dumps(json.loads(text)) if text.startswith(("{", "[")) else text
        for label in RETIRED:
            assert label not in payload, (label, payload[:300])
