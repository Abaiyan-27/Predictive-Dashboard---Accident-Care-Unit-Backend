"""Admin workflow and the transfer-request workflow, against the real API and a
migrated database (PostgREST), with the real models.

Hospitals: A = DGH – Vavuniya, B = DGH – Mannar, C = DGH – Kilinochchi (all in
the network) and a non-network hospital. Each network hospital has a nurse, a
doctor and an admin; A has a second nurse; there is a government officer.
"""
from __future__ import annotations

import uuid

import pytest

from app.core.security import create_access_token

from .conftest import psycopg, running_api
from .test_doctor_workflow import CASE_A

HOSPITAL_A, HOSPITAL_B, HOSPITAL_C = "DGH – Vavuniya", "DGH – Mannar", "DGH – Kilinochchi"
OUTSIDE = "General Hospital Colombo"
DOCTOR_MODELS = ("stay_duration", "transfer", "discharge")
ORIGINAL_TABLES = ("Hospital", "User", "Government_Officer", "Patient", "Socioeconomic_Info", "Accident_Record",
                   "Injury_Description", "Treatment", "Transfer", "Hospital_Patient", "AuditLog")


@pytest.fixture(scope="module")
def world(pg_cluster):
    db = pg_cluster.create_database("admin_transfers")
    pg_cluster.migrate(db)
    with psycopg.connect(pg_cluster.dsn(db), autocommit=True) as setup:
        setup.execute(f'alter database "{db}" set timezone to \'UTC\'')
    with running_api(pg_cluster, db) as (client, conn):
        conn.execute('insert into "Hospital" (name, type) values (%s, %s)', (OUTSIDE, "General Hospital"))
        hospitals = {
            name: str(conn.execute('select hospital_id from "Hospital" where name = %s', (name,)).fetchone()[0])
            for name in (HOSPITAL_A, HOSPITAL_B, HOSPITAL_C, OUTSIDE)
        }
        people: dict[str, dict] = {}
        for key, role, hospital in [
            ("nurse_a", "nurse", HOSPITAL_A), ("nurse_a2", "nurse", HOSPITAL_A), ("doctor_a", "doctor", HOSPITAL_A),
            ("admin_a", "admin", HOSPITAL_A),
            ("nurse_b", "nurse", HOSPITAL_B), ("doctor_b", "doctor", HOSPITAL_B), ("admin_b", "admin", HOSPITAL_B),
            ("nurse_c", "nurse", HOSPITAL_C), ("doctor_c", "doctor", HOSPITAL_C), ("admin_c", "admin", HOSPITAL_C),
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
        yield {"client": client, "db": conn, "people": people, "hospitals": hospitals}


# ── Helpers ────────────────────────────────────────────────────────────────

def call(world, who, method, path, **kwargs):
    return world["client"].request(method, path, headers=world["people"][who]["headers"], **kwargs)


def admit(world, nurse="nurse_a", treat_with="nurse_a"):
    """A patient admitted by a nurse and (optionally) treated, so all doctor models have run."""
    body = call(world, nurse, "POST", "/api/patients/intake",
                json={"full_name": f"Patient {uuid.uuid4().hex[:6]}", "gender": "Female", **CASE_A["intake"]})
    assert body.status_code == 201, body.text
    accident = body.json()["accident_record"]["accident_id"]
    if treat_with:
        treated = call(world, treat_with, "POST", "/api/treatments", json={"accident_id": accident, **CASE_A["treatment"]})
        assert treated.status_code == 201, treated.text
    return accident


def request_transfer(world, accident, to=HOSPITAL_C, who="admin_a", message="Prepare the patient for transfer."):
    return call(world, who, "POST", "/api/admin/transfers",
                json={"accident_id": accident, "to_hospital_id": world["hospitals"][to], "message": message})


def requested(world, accident, **kwargs):
    response = request_transfer(world, accident, **kwargs)
    assert response.status_code == 201, response.text
    return response.json()


def authorized(world, accident, **kwargs):
    transfer = requested(world, accident, **kwargs)
    response = call(world, "admin_a", "POST", f"/api/admin/transfers/{transfer['transfer_id']}/authorize",
                    json={"note": "Bed confirmed at the destination"})
    assert response.status_code == 200, response.text
    return response.json()


def process(world, transfer_id, who="nurse_a", **body):
    return call(world, who, "POST", f"/api/patients/transfer-requests/{transfer_id}/process", json=body)


def row(world, transfer_id):
    cursor = world["db"].execute('select * from "Transfer" where transfer_id = %s', (transfer_id,))
    names = [column.name for column in cursor.description]
    return dict(zip(names, cursor.fetchone()))


def stays(world, accident):
    return world["db"].execute(
        'select hospital_id::text, entered_at, exited_at from "Hospital_Patient" where accident_id = %s order by entered_at, exited_at nulls last',
        (accident,)).fetchall()


def audit(world, action, accident=None, hospital=None):
    query = 'select actor_id::text, role, hospital_id::text, details from "AuditLog" where action_type = %s'
    params: list = [action]
    if accident:
        query += " and accident_id = %s"
        params.append(accident)
    if hospital:
        query += " and hospital_id = %s"
        params.append(hospital)
    return world["db"].execute(query + ' order by "timestamp"', params).fetchall()


TRANSFERRED = {"outcome": "transferred", "state_of_transfer": "cervical collar used during transfer"}


# ── The complete flow ─────────────────────────────────────────────────────

def test_admin_request_to_nurse_processing_end_to_end(world):
    people, hospitals = world["people"], world["hospitals"]
    accident = admit(world)

    transfer = requested(world, accident, message="Prepare Patient A for transfer to Kilinochchi.")
    tid = transfer["transfer_id"]
    assert transfer["status"] == "pending_authorization" and transfer["allowed_actions"] == ["authorize", "reject"]
    assert transfer["request_message"] == "Prepare Patient A for transfer to Kilinochchi."
    stored = row(world, tid)
    assert (str(stored["from_hospital_id"]), str(stored["to_hospital_id"]), str(stored["requested_by"])) == (
        hospitals[HOSPITAL_A], hospitals[HOSPITAL_C], people["admin_a"]["id"])
    assert stored["requested_at"] is not None

    # Every nurse of the sending hospital sees it, though neither registered nor requested it.
    for nurse in ("nurse_a", "nurse_a2"):
        listed = {t["transfer_id"]: t for t in call(world, nurse, "GET", "/api/patients/transfer-requests").json()}
        assert listed[tid]["request_message"].startswith("Prepare Patient A")
        assert listed[tid]["requested_by_name"] == "admin_a" and listed[tid]["destination_hospital"] == HOSPITAL_C
        assert listed[tid]["allowed_actions"] == []          # waits for the admin's authorization
    assert tid not in {t["transfer_id"] for t in call(world, "nurse_b", "GET", "/api/patients/transfer-requests").json()}
    assert process(world, tid, **TRANSFERRED).status_code == 409

    response = call(world, "admin_a", "POST", f"/api/admin/transfers/{tid}/authorize", json={"note": "Bed confirmed"})
    assert response.status_code == 200, response.text
    stored = row(world, tid)
    assert stored["status"] == "authorized" and str(stored["authorized_by"]) == people["admin_a"]["id"]
    assert stored["authorized_at"] >= stored["requested_at"] and stored["authorization_reason"] == "Bed confirmed"
    listed = {t["transfer_id"]: t for t in call(world, "nurse_a2", "GET", "/api/patients/transfer-requests").json()}
    assert listed[tid]["allowed_actions"] == ["complete", "not_transferred"]

    response = process(world, tid, who="nurse_a2", note="Left by ambulance at 10:40", **TRANSFERRED)
    assert response.status_code == 200, response.text
    stored = row(world, tid)
    assert stored["status"] == "approved"
    assert str(stored["processed_by"]) == people["nurse_a2"]["id"] and stored["processed_at"] is not None
    assert stored["decided_at"] == stored["processed_at"] and stored["approved_by"] is None
    assert stored["rejected_by"] is None and stored["state_of_transfer"] == "cervical collar used during transfer"

    # The destination now holds the episode; the old stay is closed, not removed.
    holder = world["db"].execute('select hospital_id::text from "Accident_Record" where accident_id = %s',
                                 (accident,)).fetchone()[0]
    assert holder == hospitals[HOSPITAL_C]
    history = stays(world, accident)
    assert [h[0] for h in history] == [hospitals[HOSPITAL_A], hospitals[HOSPITAL_C]]
    assert history[0][2] == history[1][1] == stored["processed_at"]    # handover at one instant
    assert history[1][2] is None
    assert sum(1 for h in history if h[2] is None) == 1

    # The audit answers who requested it and who processed it.
    [created] = audit(world, "transfer_requested", accident)
    assert created[0] == people["admin_a"]["id"] and created[1] == "admin"
    assert created[3]["new_status"] == "pending_authorization" and created[3]["message"].startswith("Prepare Patient A")
    [done] = audit(world, "transfer_completed", accident)
    assert (done[0], done[1], done[2]) == (people["nurse_a2"]["id"], "nurse", hospitals[HOSPITAL_A])
    details = done[3]
    assert (details["old_status"], details["new_status"]) == ("authorized", "approved")
    assert details["requested_by"] == people["admin_a"]["id"] and details["requested_by_name"] == "admin_a"
    assert details["processed_by"] == people["nurse_a2"]["id"] and details["processed_by_name"] == "nurse_a2"
    assert details["authorized_by_name"] == "admin_a" and details["rejected_by_name"] is None
    assert (details["from_hospital"], details["to_hospital"]) == (HOSPITAL_A, HOSPITAL_C)
    assert audit(world, "transfer_received", accident, hospital=hospitals[HOSPITAL_C])
    [authorized_entry] = audit(world, "transfer_authorized", accident)
    assert authorized_entry[0] == people["admin_a"]["id"]

    # The receiving admin sees the arrival in their own log; the sending admin lost access to the episode.
    c_log = call(world, "admin_c", "GET", "/api/admin/audit-logs", params={"accident_id": accident}).json()
    assert {entry["action_type"] for entry in c_log} == {"transfer_received"}
    assert call(world, "admin_a", "GET", f"/api/admin/episodes/{accident}").status_code == 404
    assert call(world, "admin_c", "GET", f"/api/admin/episodes/{accident}").status_code == 200

    # Final means final.
    assert process(world, tid, **TRANSFERRED).status_code in (403, 409)
    assert call(world, "admin_a", "POST", f"/api/admin/transfers/{tid}/reject", json={"reason": "x"}).status_code == 409
    assert call(world, "admin_a", "POST", f"/api/admin/transfers/{tid}/authorize", json={}).status_code == 409


# ── Request validation ────────────────────────────────────────────────────

def test_destination_must_be_another_network_hospital(world):
    accident = admit(world, treat_with=None)
    assert request_transfer(world, accident, to=HOSPITAL_A).status_code == 422
    assert request_transfer(world, accident, to=OUTSIDE).status_code == 422
    response = call(world, "admin_a", "POST", "/api/admin/transfers", json={
        "accident_id": accident, "to_hospital_id": str(uuid.uuid4()), "message": "m"})
    assert response.status_code == 422
    assert call(world, "admin_a", "POST", "/api/admin/transfers", json={
        "accident_id": accident, "to_hospital_id": world["hospitals"][HOSPITAL_B], "message": ""}).status_code == 422
    assert world["db"].execute('select count(*) from "Transfer" where accident_id = %s', (accident,)).fetchone()[0] == 0
    hospitals = call(world, "admin_a", "GET", "/api/admin/network-hospitals").json()
    names = {h["name"] for h in hospitals}
    assert len(names) == 7 and HOSPITAL_A not in names and OUTSIDE not in names


def test_origin_is_always_the_admins_hospital(world):
    accident = admit(world, treat_with=None)
    response = call(world, "admin_a", "POST", "/api/admin/transfers", json={
        "accident_id": accident, "from_hospital_id": world["hospitals"][HOSPITAL_B],
        "to_hospital_id": world["hospitals"][HOSPITAL_C], "message": "m"})
    assert response.status_code == 403
    other = admit(world, nurse="nurse_b", treat_with=None)
    assert request_transfer(world, other).status_code == 404


def test_only_one_open_transfer_per_episode(world):
    accident = admit(world, treat_with=None)
    requested(world, accident)
    assert request_transfer(world, accident, to=HOSPITAL_B).status_code == 409


# ── State machine ─────────────────────────────────────────────────────────

def test_rejection_records_the_rejecting_admin_and_is_final(world):
    accident = admit(world, treat_with=None)
    tid = requested(world, accident)["transfer_id"]
    assert call(world, "admin_a", "POST", f"/api/admin/transfers/{tid}/reject", json={}).status_code == 422
    response = call(world, "admin_a", "POST", f"/api/admin/transfers/{tid}/reject", json={"reason": "Bed no longer needed"})
    assert response.status_code == 200, response.text
    stored = row(world, tid)
    assert stored["status"] == "rejected" and str(stored["rejected_by"]) == world["people"]["admin_a"]["id"]
    assert stored["rejection_reason"] == "Bed no longer needed" and stored["decided_at"] is not None
    assert stored["approved_by"] is None and stored["processed_by"] is None
    assert call(world, "admin_a", "POST", f"/api/admin/transfers/{tid}/authorize", json={}).status_code == 409
    assert process(world, tid, **TRANSFERRED).status_code == 409
    [entry] = audit(world, "transfer_rejected", accident)
    assert (entry[3]["old_status"], entry[3]["new_status"], entry[3]["note"]) == (
        "pending_authorization", "rejected", "Bed no longer needed")
    assert entry[3]["rejected_by_name"] == "admin_a" and entry[3]["requested_by_name"] == "admin_a"
    # A closed request does not block a new one.
    assert request_transfer(world, accident).status_code == 201


def test_an_authorized_transfer_can_still_be_rejected_by_the_admin(world):
    tid = authorized(world, admit(world, treat_with=None))["transfer_id"]
    assert call(world, "admin_a", "POST", f"/api/admin/transfers/{tid}/reject", json={"reason": "Destination full"}).status_code == 200
    assert process(world, tid, **TRANSFERRED).status_code == 409


def test_nurse_records_a_transfer_that_did_not_happen(world):
    accident = admit(world, treat_with=None)
    before = stays(world, accident)
    tid = authorized(world, accident)["transfer_id"]
    assert process(world, tid, outcome="not_transferred").status_code == 422          # a reason is required
    response = process(world, tid, outcome="not_transferred", note="Patient refused transfer")
    assert response.status_code == 200, response.text
    stored = row(world, tid)
    assert stored["status"] == "rejected" and str(stored["processed_by"]) == world["people"]["nurse_a"]["id"]
    assert stored["processing_note"] == "Patient refused transfer" and stored["rejected_by"] is None
    assert stays(world, accident) == before
    assert world["db"].execute('select hospital_id::text from "Accident_Record" where accident_id = %s',
                               (accident,)).fetchone()[0] == world["hospitals"][HOSPITAL_A]
    [entry] = audit(world, "transfer_not_completed", accident)
    assert entry[1] == "nurse" and entry[3]["new_status"] == "rejected"
    assert process(world, tid, **TRANSFERRED).status_code == 409


@pytest.mark.parametrize("body", [
    {"outcome": "transferred"},
    {"outcome": "transferred", "state_of_transfer": "by helicopter"},
    {"outcome": "lost"},
])
def test_processing_input_is_validated(world, body):
    tid = authorized(world, admit(world, treat_with=None))["transfer_id"]
    assert process(world, tid, **body).status_code == 422
    assert row(world, tid)["status"] == "authorized"


def test_only_the_sending_hospitals_admin_and_nurses_act(world):
    accident = admit(world, treat_with=None)
    tid = requested(world, accident)["transfer_id"]
    # The receiving hospital can see it, but not act on it.
    assert tid in {t["transfer_id"] for t in call(world, "admin_c", "GET", "/api/admin/transfers").json()}
    assert call(world, "admin_c", "POST", f"/api/admin/transfers/{tid}/authorize", json={}).status_code == 403
    # Uninvolved hospitals cannot even see it.
    assert call(world, "admin_b", "POST", f"/api/admin/transfers/{tid}/authorize", json={}).status_code == 404
    # Doctors, nurses and government cannot perform admin actions.
    for who in ("doctor_a", "nurse_a", "government"):
        assert call(world, who, "POST", f"/api/admin/transfers/{tid}/authorize", json={}).status_code == 403
        assert call(world, who, "POST", f"/api/admin/transfers/{tid}/reject", json={"reason": "x"}).status_code == 403
    assert call(world, "admin_a", "POST", f"/api/admin/transfers/{tid}/authorize", json={}).status_code == 200
    # Only a sending-hospital nurse processes it.
    for who, expected in (("nurse_c", 403), ("nurse_b", 404), ("admin_a", 403), ("doctor_a", 403), ("government", 403)):
        assert process(world, tid, who=who, **TRANSFERRED).status_code == expected, who
    assert row(world, tid)["status"] == "authorized"


def test_the_database_enforces_the_state_machine(world):
    db = world["db"]
    accident = admit(world, treat_with=None)
    tid = requested(world, accident)["transfer_id"]
    with pytest.raises(psycopg.Error) as invalid:
        with db.transaction():
            db.execute('update "Transfer" set status = %s where transfer_id = %s', ("approved", tid))
    assert invalid.value.sqlstate == "PT409"
    with pytest.raises(psycopg.Error) as outside:
        with db.transaction():
            db.execute('insert into "Transfer" (accident_id, from_hospital_id, to_hospital_id, requested_by) '
                       "values (%s, %s, %s, %s)", (accident, world["hospitals"][HOSPITAL_A],
                                                  world["hospitals"][OUTSIDE], world["people"]["admin_a"]["id"]))
    assert outside.value.sqlstate == "PT422"
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db.transaction():
            db.execute('insert into "Transfer" (accident_id, from_hospital_id, to_hospital_id, requested_by) '
                       "values (%s, %s, %s, %s)", (accident, world["hospitals"][HOSPITAL_A],
                                                  world["hospitals"][HOSPITAL_B], world["people"]["admin_a"]["id"]))
    # The function re-checks the actor: a nurse cannot authorize, even calling it directly.
    with pytest.raises(psycopg.Error) as wrong_actor:
        with db.transaction():
            db.execute("select * from transfer_transition(%s, 'authorize', %s)", (tid, world["people"]["nurse_a"]["id"]))
    assert wrong_actor.value.sqlstate == "PT403"
    assert row(world, tid)["status"] == "pending_authorization"


# ── Prediction freshness across a transfer ────────────────────────────────

def test_transfer_makes_predictions_stale_without_touching_history_or_decisions(world):
    people = world["people"]
    accident = admit(world)
    episode = call(world, "doctor_a", "GET", f"/api/doctor/accidents/{accident}").json()
    before = {key: episode["models"][key]["prediction"]["prediction_id"] for key in DOCTOR_MODELS}
    stay_id = before["stay_duration"]
    assert call(world, "doctor_a", "POST", f"/api/doctor/accidents/{accident}/predictions/{stay_id}/decision",
                json={"action": "accept"}).status_code == 201
    old_rows = world["db"].execute(
        'select prediction_id::text, model_key, prediction from "Model_Prediction" where accident_id = %s order by prediction_seq',
        (accident,)).fetchall()
    old_decision = world["db"].execute(
        'select decision_id, final_decision, decision_source, decided_at from "Prediction_Decision" where prediction_id = %s',
        (stay_id,)).fetchall()

    # An open request makes the transfer outcome unknown: Discharge reads it, so its prediction is stale.
    tid = authorized(world, accident)["transfer_id"]
    models = call(world, "doctor_a", "GET", f"/api/doctor/accidents/{accident}").json()["models"]
    assert models["discharge"]["prediction"] is None or models["discharge"]["prediction"]["stale"] is True
    assert models["stay_duration"]["state"] == "ACCEPTED"           # the decision stands

    # Admins can view predictions but never decide on them.
    for key in DOCTOR_MODELS:
        assert call(world, "admin_a", "POST", f"/api/doctor/accidents/{accident}/predictions/{before[key]}/decision",
                    json={"action": "accept"}).status_code == 403
    assert call(world, "admin_a", "POST", f"/api/doctor/accidents/{accident}/run").status_code == 403

    assert process(world, tid, **TRANSFERRED).status_code == 200

    # At the destination, every earlier prediction belongs to the old hospital and inputs.
    episode = call(world, "doctor_c", "GET", f"/api/doctor/accidents/{accident}").json()
    for key in ("transfer", "discharge"):
        assert episode["models"][key]["prediction"]["stale"] is True
        assert episode["models"][key]["state"] in {"NOT_RUN", "WAITING_FOR_INPUTS"}
    assert episode["models"]["stay_duration"]["state"] == "ACCEPTED"
    response = call(world, "doctor_c", "POST", f"/api/doctor/accidents/{accident}/run")
    assert response.status_code == 200, response.text
    fresh = response.json()["models"]
    for key in ("transfer", "discharge"):
        prediction = fresh[key]["prediction"]
        assert prediction["stale"] is False and prediction["prediction_id"] != before[key]
        assert prediction["hospital_id"] == world["hospitals"][HOSPITAL_C]
    inputs = world["db"].execute(
        'select inputs from "Model_Prediction" where prediction_id = %s', (fresh["discharge"]["prediction"]["prediction_id"],)
    ).fetchone()[0]
    assert inputs["Transfer To Next Hospital"] == "yes"
    assert inputs["State of Transfer"] == "cervical collar used during transfer"

    # History kept, decision unchanged, and no decision was created by anyone else.
    rows = world["db"].execute(
        'select prediction_id::text, model_key, prediction from "Model_Prediction" where accident_id = %s order by prediction_seq',
        (accident,)).fetchall()
    assert rows[:len(old_rows)] == old_rows
    assert world["db"].execute(
        'select decision_id, final_decision, decision_source, decided_at from "Prediction_Decision" where prediction_id = %s',
        (stay_id,)).fetchall() == old_decision
    decided_by = world["db"].execute(
        'select distinct d.decided_by::text from "Prediction_Decision" d join "Model_Prediction" p using (prediction_id) '
        "where p.accident_id = %s", (accident,)).fetchall()
    assert decided_by == [(people["doctor_a"]["id"],)]


# ── Admin views ───────────────────────────────────────────────────────────

def test_admin_patient_list_and_episode_are_read_only_and_scoped(world):
    accident = admit(world)
    other = admit(world, nurse="nurse_b", treat_with="nurse_b")
    listed = {row["accident_id"]: row for row in call(world, "admin_a", "GET", "/api/admin/patients").json()}
    assert accident in listed and other not in listed
    entry = listed[accident]
    assert entry["patient"]["full_name"].startswith("Patient")
    assert entry["stay_duration"]["prediction"]["prediction"] == "2-3 days"
    assert entry["transfer"]["prediction"]["prediction"] == "No transfer"
    assert "discharge" not in entry and "severity" not in entry

    episode = call(world, "admin_a", "GET", f"/api/admin/episodes/{accident}").json()
    assert set(episode["predictions"]) == {"stay_duration", "transfer"}
    for view in episode["predictions"].values():
        assert set(view) == {"title", "state", "prediction"}       # no decision controls or flags
    assert episode["episode"]["hospital_name"] == HOSPITAL_A and episode["can_request_transfer"] is True
    assert [stay["hospital_name"] for stay in episode["stays"]] == [HOSPITAL_A]
    assert call(world, "admin_a", "GET", f"/api/admin/episodes/{other}").status_code == 404

    # The doctor-only prognosis endpoint stays closed to admins.
    patient = world["db"].execute('select patient_id::text from "Accident_Record" where accident_id = %s', (accident,)).fetchone()[0]
    assert call(world, "admin_a", "GET", f"/api/patients/{patient}/prognosis").status_code == 403


@pytest.mark.parametrize("path, body", [
    ("/api/severity/accidents/{accident}/run", None),
    ("/api/doctor/accidents/{accident}/run", None),
    ("/api/treatments", {"accident_id": "{accident}", **CASE_A["treatment"]}),
])
def test_admin_cannot_run_models_or_record_treatment(world, path, body):
    accident = admit(world)
    payload = {k: (v.format(accident=accident) if isinstance(v, str) else v) for k, v in (body or {}).items()}
    assert call(world, "admin_a", "POST", path.format(accident=accident), json=payload or None).status_code == 403


def test_admin_cannot_decide_severity_or_edit_clinical_inputs(world):
    accident = admit(world, treat_with=None)
    severity = call(world, "nurse_a", "POST", f"/api/severity/accidents/{accident}/run").json()
    prediction_id = severity["prediction"]["prediction_id"]
    assert call(world, "admin_a", "POST", f"/api/severity/predictions/{prediction_id}/decision",
                json={"action": "accept"}).status_code == 403
    assert call(world, "admin_a", "PATCH", f"/api/patients/intake/{accident}", json={"age": "60"}).status_code == 403
    assert world["db"].execute('select count(*) from "Prediction_Decision" where prediction_id = %s',
                               (prediction_id,)).fetchone()[0] == 0


# ── Users ─────────────────────────────────────────────────────────────────

class FakeAuthAdmin:
    def __init__(self):
        self.created = []

    def create_user(self, attributes):
        self.created.append(attributes)
        return type("Result", (), {"user": type("AuthUser", (), {"id": str(uuid.uuid4())})()})()

    def list_users(self, **kwargs):
        return []


@pytest.fixture
def fake_auth(monkeypatch):
    """Supabase Auth is not part of the local harness; only it is faked, the database is real."""
    from app.db.session import get_supabase
    from app.main import app
    supabase = app.dependency_overrides[get_supabase]()
    admin = FakeAuthAdmin()
    monkeypatch.setattr(supabase, "auth", type("Auth", (), {"admin": admin})())
    return admin


@pytest.mark.parametrize("role", ["nurse", "doctor", "admin"])
def test_admin_creates_staff_for_their_own_hospital(world, fake_auth, role):
    email = f"new-{role}-{uuid.uuid4().hex[:6]}@example.org"
    response = call(world, "admin_a", "POST", "/api/admin/users", json={"name": f"New {role}", "email": email, "role": role})
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["hospital_id"] == world["hospitals"][HOSPITAL_A] and body["role"] == role
    assert body["temporary_password"] and body["must_change_password"] is True
    assert fake_auth.created[-1]["user_metadata"] == {"must_change_password": True}
    stored = world["db"].execute('select role, hospital_id::text from "User" where email = %s', (email,)).fetchone()
    assert stored == (role, world["hospitals"][HOSPITAL_A])
    [entry] = [e for e in audit(world, "user_created") if e[3] and e[3].get("email") == email]
    assert entry[0] == world["people"]["admin_a"]["id"] and entry[3]["role"] == role


def test_admin_cannot_create_users_elsewhere_or_take_over_logins(world, fake_auth):
    hospitals = world["hospitals"]
    assert call(world, "admin_a", "POST", "/api/admin/users", json={
        "name": "X", "email": "x-b@example.org", "role": "nurse", "hospital_id": hospitals[HOSPITAL_B]}).status_code == 403
    assert call(world, "admin_a", "POST", "/api/admin/users", json={
        "name": "X", "email": "x-g@example.org", "role": "government"}).status_code == 422
    for taken in ("officer@example.org", "NURSE_B@example.org"):
        assert call(world, "admin_a", "POST", "/api/admin/users", json={
            "name": "X", "email": taken, "role": "nurse"}).status_code == 409
    assert fake_auth.created == []


def test_users_are_listed_per_hospital(world):
    names = {user["name"] for user in call(world, "admin_a", "GET", "/api/admin/users").json()}
    assert {"nurse_a", "nurse_a2", "doctor_a", "admin_a"} <= names
    assert not names & {"nurse_b", "admin_b", "admin_c"}


# ── Audit log ─────────────────────────────────────────────────────────────

def test_audit_log_is_scoped_filtered_paginated_and_named(world):
    accident = admit(world)
    requested(world, accident)
    log = call(world, "admin_a", "GET", "/api/admin/audit-logs", params={"action": "transfer_*"}).json()
    assert log and all(entry["action_type"].startswith("transfer_") for entry in log)
    assert all(entry["hospital_id"] == world["hospitals"][HOSPITAL_A] for entry in log)
    mine = next(e for e in log if e["accident_id"] == accident and e["action_type"] == "transfer_requested")
    assert mine["actor_name"] == "admin_a" and mine["hospital_name"] == HOSPITAL_A
    assert mine["patient_name"].startswith("Patient") and mine["details"]["to_hospital"] == HOSPITAL_C

    episode_log = call(world, "admin_a", "GET", "/api/admin/audit-logs", params={"accident_id": accident}).json()
    assert {"patient_intake", "treatment_logged", "transfer_requested"} <= {e["action_type"] for e in episode_log}
    first = call(world, "admin_a", "GET", "/api/admin/audit-logs", params={"limit": 2}).json()
    second = call(world, "admin_a", "GET", "/api/admin/audit-logs", params={"limit": 2, "offset": 2}).json()
    assert len(first) == 2 and not {e["log_id"] for e in first} & {e["log_id"] for e in second}

    # Hospital B's admin never sees Hospital A's entries, even when filtering by A's episode.
    assert call(world, "admin_b", "GET", "/api/admin/audit-logs", params={"accident_id": accident}).json() == []
    assert all(e["hospital_id"] == world["hospitals"][HOSPITAL_B]
               for e in call(world, "admin_b", "GET", "/api/admin/audit-logs").json())


# ── Government ────────────────────────────────────────────────────────────

def test_government_has_no_admin_or_transfer_access(world):
    accident = admit(world, treat_with=None)
    tid = requested(world, accident)["transfer_id"]
    for method, path in (("GET", "/api/admin/patients"), ("GET", f"/api/admin/episodes/{accident}"),
                         ("GET", "/api/admin/transfers"), ("GET", "/api/admin/audit-logs"),
                         ("GET", "/api/admin/users"), ("GET", "/api/admin/network-hospitals"),
                         ("GET", "/api/patients/transfer-requests")):
        assert call(world, "government", method, path).status_code == 403, path
    assert request_transfer(world, accident, who="government").status_code == 403
    assert process(world, tid, who="government", **TRANSFERRED).status_code == 403


# ── RLS on the original tables (migration 202610060002) ───────────────────

def test_original_tables_are_closed_to_supabase_public_roles(world):
    db = world["db"]
    for table in ORIGINAL_TABLES:
        enabled = db.execute("select relrowsecurity from pg_class where oid = %s::regclass", (f'public."{table}"',)).fetchone()[0]
        assert enabled, table
    # Simulate Supabase's default grants: even with table privileges, anon sees no rows.
    with db.transaction():
        db.execute('grant select on "Patient", "Accident_Record", "Transfer" to anon')
        assert db.execute('select count(*) from "Patient"').fetchone()[0] > 0
        db.execute("set local role anon")
        for table in ("Patient", "Accident_Record", "Transfer"):
            assert db.execute(f'select count(*) from "{table}"').fetchone()[0] == 0, table
        db.execute("reset role")
        db.execute('revoke select on "Patient", "Accident_Record", "Transfer" from anon')
    assert db.execute("select has_function_privilege('anon', "
                      "'public.transfer_transition(uuid, text, uuid, text, text)', 'execute')").fetchone()[0] is False
