"""Phase 9 hardening, against the real API and a migrated database (PostgREST):
concurrency, database-enforced finality and input locks, transfer atomicity,
forged request bodies, RLS on direct access, and account takeover protection.
"""
from __future__ import annotations

import threading
import uuid
from types import SimpleNamespace

import pytest

from app.core.security import create_access_token

from .conftest import psycopg, running_api
from .test_doctor_workflow import CASE_A

HOSPITAL_A, HOSPITAL_B, HOSPITAL_C = "DGH – Vavuniya", "DGH – Mannar", "DGH – Kilinochchi"
TRANSITIONS_ALLOWED = {("pending_authorization", "authorized"), ("pending_authorization", "rejected"),
                       ("authorized", "approved"), ("authorized", "rejected")}
STATUSES = ("pending_authorization", "authorized", "approved", "rejected")
TRANSFERRED = {"outcome": "transferred", "state_of_transfer": "cervical collar used during transfer"}


@pytest.fixture(scope="module")
def world(pg_cluster):
    db = pg_cluster.create_database("integrity_hardening")
    pg_cluster.migrate(db)
    with psycopg.connect(pg_cluster.dsn(db), autocommit=True) as setup:
        setup.execute(f'alter database "{db}" set timezone to \'UTC\'')
    with running_api(pg_cluster, db) as (client, conn):
        hospitals = {
            name: str(conn.execute('select hospital_id from "Hospital" where name = %s', (name,)).fetchone()[0])
            for name in (HOSPITAL_A, HOSPITAL_B, HOSPITAL_C)
        }
        people: dict[str, dict] = {}
        for key, role, hospital in [
            ("nurse_a", "nurse", HOSPITAL_A), ("nurse_a2", "nurse", HOSPITAL_A), ("doctor_a", "doctor", HOSPITAL_A),
            ("doctor_a2", "doctor", HOSPITAL_A), ("admin_a", "admin", HOSPITAL_A), ("admin_a2", "admin", HOSPITAL_A),
            ("nurse_b", "nurse", HOSPITAL_B), ("admin_b", "admin", HOSPITAL_B),
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
        yield {"client": client, "db": conn, "dsn": pg_cluster.dsn(db), "people": people, "hospitals": hospitals}


# ── Helpers ────────────────────────────────────────────────────────────────

def call(world, who, method, path, **kwargs):
    return world["client"].request(method, path, headers=world["people"][who]["headers"], **kwargs)


def admit(world, nurse="nurse_a", run_severity=True):
    body = call(world, nurse, "POST", "/api/patients/intake",
                json={"full_name": f"Patient {uuid.uuid4().hex[:6]}", "gender": "Female", **CASE_A["intake"]})
    assert body.status_code == 201, body.text
    accident = body.json()["accident_record"]["accident_id"]
    prediction = None
    if run_severity:
        run = call(world, nurse, "POST", f"/api/severity/accidents/{accident}/run")
        assert run.status_code == 200, run.text
        prediction = run.json()["prediction"]
    return accident, prediction


def treat(world, accident, doctor="doctor_a"):
    response = call(world, doctor, "POST", "/api/treatments", json={"accident_id": accident, **CASE_A["treatment"]})
    assert response.status_code == 201, response.text
    view = call(world, doctor, "GET", f"/api/doctor/accidents/{accident}").json()
    return {key: view["models"][key]["prediction"] for key in ("stay_duration", "transfer", "discharge")}


def decisions(world, accident, model="severity"):
    return world["db"].execute(
        'select d.decided_by::text, d.decision_source from "Prediction_Decision" d '
        'join "Model_Prediction" p using (prediction_id) where p.accident_id = %s and d.model_key = %s',
        (accident, model)).fetchall()


def insert_decision_sql(conn, prediction_id, model_key, decision, source, user):
    conn.execute('insert into "Prediction_Decision" (prediction_id, model_key, final_decision, decision_source, '
                 'decided_by) values (%s, %s, %s, %s, %s)', (prediction_id, model_key, decision, source, user))


# ── 1. Concurrent decisions ───────────────────────────────────────────────

def test_two_parallel_database_sessions_cannot_both_decide(world):
    """Real concurrency: the second session waits on the episode lock, then is refused."""
    accident, prediction = admit(world)
    first = psycopg.connect(world["dsn"])
    second = psycopg.connect(world["dsn"])
    outcome: dict[str, object] = {}
    try:
        insert_decision_sql(first, prediction["prediction_id"], "severity", prediction["prediction"], "MODEL",
                            world["people"]["nurse_a"]["id"])           # holds the lock, not yet committed

        def other_nurse():
            try:
                insert_decision_sql(second, prediction["prediction_id"], "severity",
                                    "LOW" if prediction["prediction"] != "LOW" else "HIGH", "NURSE",
                                    world["people"]["nurse_a2"]["id"])
                second.commit()
                outcome["result"] = "committed"
            except psycopg.Error as exc:
                second.rollback()
                outcome["result"] = exc.sqlstate

        thread = threading.Thread(target=other_nurse)
        thread.start()
        thread.join(timeout=1.5)
        assert thread.is_alive(), "the second decision should be waiting for the first transaction"
        first.commit()
        thread.join(timeout=10)
        assert outcome["result"] == "23505"
    finally:
        first.close()
        second.close()
    assert decisions(world, accident) == [(world["people"]["nurse_a"]["id"], "MODEL")]


def test_the_losing_api_request_gets_a_clean_409(world, monkeypatch):
    """Simulate the race window: the API's own check passes, the database still refuses."""
    accident, prediction = admit(world)
    insert_decision_sql(world["db"], prediction["prediction_id"], "severity", prediction["prediction"], "MODEL",
                        world["people"]["nurse_a2"]["id"])
    from app.services import severity_workflow
    monkeypatch.setattr(severity_workflow, "latest_decision", lambda history, decisions: None)
    response = call(world, "nurse_a", "POST", f"/api/severity/predictions/{prediction['prediction_id']}/decision",
                    json={"action": "accept"})
    assert response.status_code == 409 and "already been recorded" in response.json()["detail"]
    assert len(decisions(world, accident)) == 1
    stored = world["db"].execute('select prediction, status from "Model_Prediction" where prediction_id = %s',
                                 (prediction["prediction_id"],)).fetchone()
    assert stored == (prediction["prediction"], prediction["status"])     # the prediction is untouched


def test_parallel_doctor_decisions_on_one_model_leave_one_decision(world):
    accident, _ = admit(world)
    stay = treat(world, accident)["stay_duration"]
    path = f"/api/doctor/accidents/{accident}/predictions/{stay['prediction_id']}/decision"
    results: list[int] = []
    barrier = threading.Barrier(2)

    def decide(who, body):
        barrier.wait()
        results.append(call(world, who, "POST", path, json=body).status_code)

    other = next(label for label in ("0-1 days", "2-3 days", "4+ days") if label != stay["prediction"])
    threads = [threading.Thread(target=decide, args=("doctor_a", {"action": "accept"})),
               threading.Thread(target=decide, args=("doctor_a2", {"action": "override", "final_decision": other}))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert sorted(results) == [201, 409]
    assert len(decisions(world, accident, "stay_duration")) == 1


def test_a_decision_on_an_older_prediction_is_refused_by_the_database(world):
    accident, first = admit(world, run_severity=True)
    edited = call(world, "nurse_a", "PATCH", f"/api/patients/intake/{accident}", json={"helmet_worn": "yes"})
    assert edited.status_code == 200, edited.text
    rerun = call(world, "nurse_a", "POST", f"/api/severity/accidents/{accident}/run").json()["prediction"]
    assert rerun["prediction_id"] != first["prediction_id"]
    with pytest.raises(psycopg.errors.CheckViolation, match="newer"):
        insert_decision_sql(world["db"], first["prediction_id"], "severity", first["prediction"], "MODEL",
                            world["people"]["nurse_a"]["id"])


def test_decisions_cannot_be_updated_deleted_or_replaced(world):
    accident, prediction = admit(world)
    assert call(world, "nurse_a", "POST", f"/api/severity/predictions/{prediction['prediction_id']}/decision",
                json={"action": "accept"}).status_code == 201
    db = world["db"]
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        db.execute('update "Prediction_Decision" set reason = %s where prediction_id = %s', ("x", prediction["prediction_id"]))
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        db.execute('delete from "Prediction_Decision" where prediction_id = %s', (prediction["prediction_id"],))
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        db.execute('update "Model_Prediction" set prediction = %s where prediction_id = %s', ("LOW", prediction["prediction_id"]))
    # No API route updates or deletes decisions or predictions.
    from fastapi.routing import APIRoute
    from app.main import app
    for route in app.routes:
        if isinstance(route, APIRoute) and ("decision" in route.path or "prediction" in route.path):
            assert route.methods <= {"GET", "POST"}, route.path


# ── 2. Input locks in the database ───────────────────────────────────────

def test_locked_severity_inputs_cannot_be_changed_even_directly(world):
    accident, prediction = admit(world)
    assert call(world, "nurse_a", "POST", f"/api/severity/predictions/{prediction['prediction_id']}/decision",
                json={"action": "accept"}).status_code == 201
    db = world["db"]
    for statement, params in [
        ('update "Accident_Record" set helmet_worn = %s where accident_id = %s', ("yes", accident)),
        ('update "Patient" set age = %s where patient_id = (select patient_id from "Accident_Record" where accident_id = %s)', ("70", accident)),
        ('update "Injury_Description" set site_of_injury = %s where accident_id = %s and injury_order = 1', ("hand", accident)),
        ('delete from "Injury_Description" where accident_id = %s and injury_order = 1', (accident,)),
        ('insert into "Injury_Description" (accident_id, injury_order, site_of_injury) values (%s, 2, %s)', (accident, "knee")),
    ]:
        with pytest.raises(psycopg.errors.CheckViolation):
            db.execute(statement, params)
    # Basic details, unchanged re-saves and doctor-model inputs (before treatment) stay writable.
    db.execute('update "Patient" set full_name = %s where patient_id = (select patient_id from "Accident_Record" where accident_id = %s)', ("Renamed", accident))
    db.execute('update "Accident_Record" set helmet_worn = helmet_worn, road_type = %s where accident_id = %s', ("junction", accident))


def test_doctor_model_inputs_and_treatment_lock_in_the_database(world):
    accident, _ = admit(world)
    predictions = treat(world, accident)
    db = world["db"]
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute('update "Accident_Record" set speed = %s where accident_id = %s', ("80 - 100 km/h", accident))
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute('update "Injury_Description" set type_of_injury = %s where accident_id = %s and injury_order = 1', ("fracture", accident))
    # A second treatment is allowed until a doctor decides; afterwards the database refuses it.
    stay = predictions["stay_duration"]
    assert call(world, "doctor_a", "POST", f"/api/doctor/accidents/{accident}/predictions/{stay['prediction_id']}/decision",
                json={"action": "accept"}).status_code == 201
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute('insert into "Treatment" (accident_id, hospital_id, number_of_days_stay) values (%s, %s, 1)',
                   (accident, world["hospitals"][HOSPITAL_A]))
    response = call(world, "doctor_a", "POST", "/api/treatments", json={"accident_id": accident, **CASE_A["treatment"]})
    assert response.status_code == 409


def test_a_lock_that_appears_mid_request_is_a_409_not_a_500(world, monkeypatch):
    """The API's lock check passes (stale view); the database lock still applies."""
    accident, prediction = admit(world)
    assert call(world, "nurse_a", "POST", f"/api/severity/predictions/{prediction['prediction_id']}/decision",
                json={"action": "accept"}).status_code == 201
    from app.services import severity_workflow
    monkeypatch.setattr(severity_workflow, "lock_violations", lambda form, accident, status: [])
    response = call(world, "nurse_a", "PATCH", f"/api/patients/intake/{accident}", json={"helmet_worn": "yes"})
    assert response.status_code == 409 and "cannot be changed" in response.json()["detail"]
    assert world["db"].execute('select helmet_worn from "Accident_Record" where accident_id = %s', (accident,)).fetchone()[0] == "no"


# ── 3. Transfers ─────────────────────────────────────────────────────────

def request_transfer(world, accident, who="admin_a"):
    return call(world, who, "POST", "/api/admin/transfers",
                json={"accident_id": accident, "to_hospital_id": world["hospitals"][HOSPITAL_C], "message": "Prepare transfer."})


def authorized(world, accident):
    created = request_transfer(world, accident)
    assert created.status_code == 201, created.text
    tid = created.json()["transfer_id"]
    assert call(world, "admin_a", "POST", f"/api/admin/transfers/{tid}/authorize", json={}).status_code == 200
    return tid


def test_two_admins_requesting_at_once_create_one_open_transfer(world):
    accident, _ = admit(world, run_severity=False)
    results: list[int] = []
    barrier = threading.Barrier(2)

    def request(who):
        barrier.wait()
        results.append(request_transfer(world, accident, who=who).status_code)

    threads = [threading.Thread(target=request, args=(who,)) for who in ("admin_a", "admin_a2")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert sorted(results) == [201, 409]
    open_rows = world["db"].execute(
        "select count(*) from \"Transfer\" where accident_id = %s and status in ('pending_authorization', 'authorized')",
        (accident,)).fetchone()[0]
    assert open_rows == 1


def test_parallel_database_sessions_cannot_open_two_transfers(world):
    accident, _ = admit(world, run_severity=False)
    insert = ('insert into "Transfer" (accident_id, from_hospital_id, to_hospital_id, status, requested_by, request_message) '
              "values (%s, %s, %s, 'pending_authorization', %s, 'x')")
    params = (accident, world["hospitals"][HOSPITAL_A], world["hospitals"][HOSPITAL_C], world["people"]["admin_a"]["id"])
    first, second = psycopg.connect(world["dsn"]), psycopg.connect(world["dsn"])
    outcome = {}
    try:
        first.execute(insert, params)

        def other():
            try:
                second.execute(insert, params)
                second.commit()
                outcome["result"] = "committed"
            except psycopg.Error as exc:
                second.rollback()
                outcome["result"] = exc.sqlstate

        thread = threading.Thread(target=other)
        thread.start()
        thread.join(timeout=1.5)
        first.commit()
        thread.join(timeout=10)
        assert outcome["result"] == "23505"
    finally:
        first.close()
        second.close()


def test_a_failure_inside_the_transfer_transaction_rolls_everything_back(world):
    accident, _ = admit(world, run_severity=False)
    tid = authorized(world, accident)
    db = world["db"]
    before = {
        "transfer": db.execute('select status, processed_by, processed_at from "Transfer" where transfer_id = %s', (tid,)).fetchone(),
        "hospital": db.execute('select hospital_id::text from "Accident_Record" where accident_id = %s', (accident,)).fetchone()[0],
        "stays": db.execute('select hospital_id::text, exited_at from "Hospital_Patient" where accident_id = %s order by entered_at', (accident,)).fetchall(),
    }
    # Make the last step of the transaction (opening the new stay) fail.
    db.execute(f"""
        create or replace function pg_temp_fail_stay() returns trigger language plpgsql as $$
        begin
            if new.accident_id = '{accident}'::uuid then raise exception 'forced failure for the rollback test'; end if;
            return new;
        end $$;
        create trigger fail_new_stay before insert on "Hospital_Patient"
            for each row execute function pg_temp_fail_stay();
    """)
    try:
        try:
            response = call(world, "nurse_a", "POST", f"/api/patients/transfer-requests/{tid}/process", json=TRANSFERRED)
            assert response.status_code >= 400
        except Exception:
            pass          # the test client re-raises unexpected server errors; either way it failed
    finally:
        db.execute('drop trigger fail_new_stay on "Hospital_Patient"; drop function pg_temp_fail_stay()')
    after = {
        "transfer": db.execute('select status, processed_by, processed_at from "Transfer" where transfer_id = %s', (tid,)).fetchone(),
        "hospital": db.execute('select hospital_id::text from "Accident_Record" where accident_id = %s', (accident,)).fetchone()[0],
        "stays": db.execute('select hospital_id::text, exited_at from "Hospital_Patient" where accident_id = %s order by entered_at', (accident,)).fetchall(),
    }
    assert after == before
    assert after["transfer"][0] == "authorized" and after["hospital"] == world["hospitals"][HOSPITAL_A]
    assert len(after["stays"]) == 1 and after["stays"][0][1] is None
    assert not db.execute("select 1 from \"AuditLog\" where action_type = 'transfer_completed' and accident_id = %s", (accident,)).fetchall()
    # And the same request succeeds once the fault is gone.
    assert call(world, "nurse_a", "POST", f"/api/patients/transfer-requests/{tid}/process", json=TRANSFERRED).status_code == 200


def _transfer_in(world, status):
    accident, _ = admit(world, run_severity=False)
    if status == "pending_authorization":
        return request_transfer(world, accident).json()["transfer_id"]
    tid = authorized(world, accident)
    if status == "approved":
        assert call(world, "nurse_a", "POST", f"/api/patients/transfer-requests/{tid}/process", json=TRANSFERRED).status_code == 200
    if status == "rejected":
        assert call(world, "admin_a", "POST", f"/api/admin/transfers/{tid}/reject", json={"reason": "No bed"}).status_code == 200
    return tid


@pytest.mark.parametrize("current", STATUSES)
def test_every_status_transition_is_enforced_for_direct_sql(world, current):
    tid = _transfer_in(world, current)
    db = world["db"]
    for target in STATUSES:
        if target == current:
            continue
        allowed = (current, target) in TRANSITIONS_ALLOWED
        try:
            with db.transaction(force_rollback=True):
                db.execute('update "Transfer" set status = %s where transfer_id = %s', (target, tid))
            refused = None
        except psycopg.Error as exc:
            refused = exc.sqlstate
        assert (refused is None) == allowed, (current, target, refused)
        if not allowed:
            assert refused == "PT409", (current, target, refused)


@pytest.mark.parametrize("current", STATUSES)
def test_the_api_offers_and_accepts_only_valid_actions(world, current):
    tid = _transfer_in(world, current)
    final = current in ("approved", "rejected")
    for who, path, body in [("admin_a", "authorize", {}), ("admin_a", "reject", {"reason": "x"})]:
        status_code = call(world, who, "POST", f"/api/admin/transfers/{tid}/{path}", json=body).status_code
        valid = (current == "pending_authorization") or (current == "authorized" and path == "reject")
        assert (status_code == 200) == valid, (current, path, status_code)
        if not valid:
            assert status_code == 409
        if valid:
            return            # the transfer moved on; the rest belongs to another starting state
    process = call(world, "nurse_a", "POST", f"/api/patients/transfer-requests/{tid}/process", json=TRANSFERRED).status_code
    assert process == 409 if (final or current == "pending_authorization") else process == 200


# ── 4. Forged request bodies ─────────────────────────────────────────────

def test_identity_fields_in_request_bodies_are_ignored_or_refused(world):
    people, hospitals = world["people"], world["hospitals"]
    db = world["db"]
    forged_identity = {"created_by": people["nurse_b"]["id"], "role": "admin", "user_id": people["admin_b"]["id"]}

    created = call(world, "nurse_a", "POST", "/api/patients/intake",
                   json={"full_name": "Forged", **CASE_A["intake"], **forged_identity})
    assert created.status_code == 201, created.text
    accident = created.json()["accident_record"]["accident_id"]
    row = db.execute('select created_by::text, hospital_id::text from "Accident_Record" where accident_id = %s', (accident,)).fetchone()
    assert row == (people["nurse_a"]["id"], hospitals[HOSPITAL_A])
    assert call(world, "nurse_a", "POST", "/api/patients/intake",
                json={"full_name": "Forged", **CASE_A["intake"], "hospital_id": hospitals[HOSPITAL_B]}).status_code == 403

    run = call(world, "nurse_a", "POST", f"/api/severity/accidents/{accident}/run").json()["prediction"]
    decided = call(world, "nurse_a", "POST", f"/api/severity/predictions/{run['prediction_id']}/decision",
                   json={"action": "accept", "decided_by": people["nurse_a2"]["id"], "decision_source": "NURSE"})
    assert decided.status_code == 201, decided.text
    assert decisions(world, accident) == [(people["nurse_a"]["id"], "MODEL")]

    transfer = call(world, "admin_a", "POST", "/api/admin/transfers", json={
        "accident_id": accident, "to_hospital_id": hospitals[HOSPITAL_C], "message": "x",
        "requested_by": people["admin_b"]["id"], "status": "approved", "authorized_by": people["admin_b"]["id"]})
    assert transfer.status_code == 201, transfer.text
    tid = transfer.json()["transfer_id"]
    row = db.execute('select requested_by::text, status, authorized_by, from_hospital_id::text from "Transfer" where transfer_id = %s', (tid,)).fetchone()
    assert row == (people["admin_a"]["id"], "pending_authorization", None, hospitals[HOSPITAL_A])
    assert call(world, "admin_a", "POST", "/api/admin/transfers", json={
        "accident_id": accident, "to_hospital_id": hospitals[HOSPITAL_C], "message": "x",
        "from_hospital_id": hospitals[HOSPITAL_B]}).status_code == 403

    call(world, "admin_a", "POST", f"/api/admin/transfers/{tid}/authorize", json={"authorized_by": people["admin_b"]["id"]})
    processed = call(world, "nurse_a", "POST", f"/api/patients/transfer-requests/{tid}/process",
                     json={**TRANSFERRED, "processed_by": people["nurse_a2"]["id"]})
    assert processed.status_code == 200, processed.text
    row = db.execute('select authorized_by::text, processed_by::text from "Transfer" where transfer_id = %s', (tid,)).fetchone()
    assert row == (people["admin_a"]["id"], people["nurse_a"]["id"])

    accident2, _ = admit(world, run_severity=False)
    treated = call(world, "doctor_a", "POST", "/api/treatments", json={
        "accident_id": accident2, **CASE_A["treatment"], "logged_by": people["doctor_a2"]["id"]})
    assert treated.status_code == 201
    logged = db.execute('select logged_by::text, hospital_id::text from "Treatment" where accident_id = %s', (accident2,)).fetchone()
    assert logged == (people["doctor_a"]["id"], hospitals[HOSPITAL_A])
    assert call(world, "doctor_a", "POST", "/api/treatments", json={
        "accident_id": accident2, **CASE_A["treatment"], "decided_by": people["nurse_a"]["id"]}).status_code == 403


# ── 5. RLS on direct access with the public roles ────────────────────────

PROTECTED_TABLES = ("Hospital", "User", "Government_Officer", "Patient", "Socioeconomic_Info", "Accident_Record",
                    "Injury_Description", "Treatment", "Transfer", "Hospital_Patient", "AuditLog",
                    "ML_Model", "Model_Output_Label", "Model_Version", "Model_Prediction", "Prediction_Decision")


@pytest.mark.parametrize("role", ["anon", "authenticated"])
def test_public_roles_cannot_read_or_change_any_protected_table(world, role):
    """Even with Supabase's default table grants, RLS leaves the public roles nothing to see or change."""
    db = world["db"]
    admit(world, run_severity=True)
    db.execute(f"do $$ begin if not exists (select from pg_roles where rolname = '{role}') then create role {role} nologin; end if; end $$")
    with db.transaction(force_rollback=True):
        tables = ", ".join(f'"{t}"' for t in PROTECTED_TABLES)
        db.execute(f"grant usage on schema public to {role}")
        db.execute(f"grant select, insert, update, delete on {tables} to {role}")
        db.execute(f"set local role {role}")
        for table in PROTECTED_TABLES:
            assert db.execute(f'select count(*) from "{table}"').fetchone()[0] == 0, table
            # UPDATE and DELETE match no visible row: nothing changes.
            assert db.execute(f'update "{table}" set {_any_column(db, table)} = {_any_column(db, table)}').rowcount == 0, table
            assert db.execute(f'delete from "{table}"').rowcount == 0, table
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with db.transaction():
                db.execute('insert into "Hospital" (name) values (%s)', ("Rogue hospital",))
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with db.transaction():
                db.execute("select * from transfer_transition(gen_random_uuid(), 'complete', gen_random_uuid())")
        db.execute("reset role")
    assert db.execute('select count(*) from "Patient"').fetchone()[0] > 0      # the backend still sees everything


def _any_column(db, table):
    return db.execute("select column_name from information_schema.columns where table_schema = 'public' "
                      "and table_name = %s order by ordinal_position limit 1", (table,)).fetchone()[0]


def test_the_backend_service_role_path_still_works(world):
    """PostgREST connects as service_role, which bypasses RLS: the whole API flow ran above."""
    accident, prediction = admit(world)
    assert prediction["prediction_id"]
    assert call(world, "admin_a", "GET", f"/api/admin/episodes/{accident}").status_code == 200


# ── 6. Account takeover through admin user creation ──────────────────────

class ExistingAuthAccount:
    """Supabase Auth where the email already has an account (create_user fails)."""

    def __init__(self, account):
        self.account = account
        self.updated: list[tuple[str, dict]] = []

    def create_user(self, attributes):
        raise RuntimeError("A user with this email address has already been registered")

    def list_users(self, page=None, per_page=None):
        return [self.account] if (page or 1) == 1 else []

    def update_user_by_id(self, user_id, attributes):
        self.updated.append((user_id, attributes))

    def delete_user(self, user_id):
        raise AssertionError("an existing account must never be deleted")


@pytest.fixture
def auth_with(monkeypatch):
    from app.db.session import get_supabase
    from app.main import app
    supabase = app.dependency_overrides[get_supabase]()

    def install(account):
        admin = ExistingAuthAccount(account)
        monkeypatch.setattr(supabase, "auth", SimpleNamespace(admin=admin))
        return admin
    return install


def test_a_used_orphan_auth_account_cannot_be_taken_over(world, auth_with):
    used = SimpleNamespace(id=str(uuid.uuid4()), email="someone@example.org",
                           user_metadata={}, last_sign_in_at="2026-09-01T10:00:00Z")
    admin = auth_with(used)
    response = call(world, "admin_a", "POST", "/api/admin/users",
                    json={"name": "Taker", "email": "SOMEONE@example.org", "role": "nurse"})
    assert response.status_code == 409 and "has been used" in response.json()["detail"]
    assert admin.updated == []
    assert not world["db"].execute('select 1 from "User" where lower(email) = %s', ("someone@example.org",)).fetchall()


def test_a_flagged_account_that_was_signed_in_to_is_not_reassigned(world, auth_with):
    signed_in = SimpleNamespace(id=str(uuid.uuid4()), email="flagged@example.org",
                                user_metadata={"must_change_password": True}, last_sign_in_at="2026-09-01T10:00:00Z")
    admin = auth_with(signed_in)
    response = call(world, "admin_a", "POST", "/api/admin/users", json={"name": "X", "email": "flagged@example.org", "role": "nurse"})
    assert response.status_code == 409 and admin.updated == []


def test_an_unused_interrupted_setup_account_is_recovered(world, auth_with):
    leftover = SimpleNamespace(id=str(uuid.uuid4()), email="leftover@example.org",
                               user_metadata={"must_change_password": True}, last_sign_in_at=None)
    admin = auth_with(leftover)
    response = call(world, "admin_a", "POST", "/api/admin/users", json={"name": "Left Over", "email": "leftover@example.org", "role": "doctor"})
    assert response.status_code == 201, response.text
    assert response.json()["user_id"] == leftover.id and response.json()["temporary_password"]
    assert admin.updated and admin.updated[0][0] == leftover.id
    assert admin.updated[0][1]["user_metadata"] == {"must_change_password": True}


# ── 7. Responses carry no secrets ────────────────────────────────────────

def test_health_endpoints_expose_no_paths_or_internals(world):
    models = world["client"].get("/health/models").json()
    text = str(models)
    assert "/home/" not in text and "/tmp/" not in text and "Traceback" not in text
    for status in models.values():
        assert set(status) == {"available", "loadable", "artifact", "version", "message"}
        assert status["artifact"] is None or "/" not in status["artifact"]
    schema = world["client"].get("/health/database-schema").json()
    assert schema["ready"] is True and "service_role" not in str(schema)


def test_api_responses_never_contain_secrets(world):
    from app.core.security import settings
    accident, _ = admit(world)
    bodies = [call(world, who, "GET", path).text for who, path in [
        ("admin_a", "/api/admin/users"), ("admin_a", "/api/admin/patients"), ("admin_a", "/api/admin/audit-logs"),
        ("nurse_a", "/api/patients/my"), ("doctor_a", f"/api/doctor/accidents/{accident}"),
        ("government", "/api/government/statistics")]]
    bodies.append(call(world, "nurse_a", "GET", f"/api/admin/episodes/{accident}").text)   # an error response
    for body in bodies:
        lowered = body.lower()
        for secret in (settings.secret_key, settings.supabase_key):
            assert secret not in body
        for marker in ("password", "hashed", "access_token", "refresh_token", "traceback"):
            assert marker not in lowered, marker


def test_paged_lists_report_their_full_size(world):
    for _ in range(3):
        admit(world, run_severity=False)
    first = call(world, "nurse_a", "GET", "/api/patients/my", params={"limit": 2})
    total = int(first.headers["X-Total-Count"])
    assert len(first.json()) == 2 and total >= 3
    rest = call(world, "nurse_a", "GET", "/api/patients/my", params={"limit": 200, "offset": 2})
    assert len(rest.json()) == total - 2
    assert not {r["accident_id"] for r in first.json()} & {r["accident_id"] for r in rest.json()}
    for who, path in (("doctor_a", f"/api/patients/hospital/{world['hospitals'][HOSPITAL_A]}"),
                      ("admin_a", "/api/admin/patients"), ("admin_a", "/api/admin/transfers")):
        response = call(world, who, "GET", path, params={"limit": 1})
        assert response.status_code == 200 and len(response.json()) <= 1
        assert int(response.headers["X-Total-Count"]) >= len(response.json())
    assert call(world, "nurse_a", "GET", "/api/patients/my", params={"limit": 1000}).status_code == 422
