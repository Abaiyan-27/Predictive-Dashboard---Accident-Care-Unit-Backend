"""Government workflow: network-wide, read-only, aggregate model statistics,
against the real API and a migrated database (PostgREST), with the real models.

Hospitals A (DGH – Vavuniya) and B (DGH – Mannar) each have a nurse, a doctor
and an admin; there is one government officer with no hospital.
"""
from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from jose import jwt

from app.core.security import settings
from app.core.security import create_access_token

from .conftest import psycopg, running_api
from .test_doctor_workflow import CASE_A, CASE_B, RETIRED

HOSPITAL_A, HOSPITAL_B = "DGH – Vavuniya", "DGH – Mannar"
MODELS = ("severity", "stay_duration", "transfer", "discharge")
LABELS = {
    "severity": ["HIGH", "BORDERLINE", "LOW", "DEFER TO CLINICAL JUDGEMENT"],
    "stay_duration": ["0-1 days", "2-3 days", "4+ days"],
    "transfer": ["Transfer", "No transfer"],
    "discharge": ["Complete recovery", "Partial recovery", "Further interventions"],
}
# Every Government endpoint, as the frontend calls them.
GOVERNMENT_GETS = [
    "/api/government/statistics",
    "/api/government/accident-trends",
    "/api/government/accident-historical-trends",
    "/api/government/accident-forecast?horizon=6",
    "/api/government/association-rules?limit=5",
    *(f"/api/government/models/{model}" for model in MODELS),
]
# Endpoints backed by the database (the others read installed artifacts and
# may answer 503 where an artifact is not installed).
DATABASE_GETS = ["/api/government/statistics", "/api/government/accident-trends",
                 *(f"/api/government/models/{model}" for model in MODELS)]


@pytest.fixture(scope="module")
def world(pg_cluster):
    db = pg_cluster.create_database("government_workflow")
    pg_cluster.migrate(db)
    with psycopg.connect(pg_cluster.dsn(db), autocommit=True) as setup:
        setup.execute(f'alter database "{db}" set timezone to \'UTC\'')
    with running_api(pg_cluster, db) as (client, conn):
        hospitals = {
            name: str(conn.execute('select hospital_id from "Hospital" where name = %s', (name,)).fetchone()[0])
            for name in (HOSPITAL_A, HOSPITAL_B)
        }
        people: dict[str, dict] = {}
        for key, role, hospital in [
            ("nurse_a", "nurse", HOSPITAL_A), ("doctor_a", "doctor", HOSPITAL_A), ("admin_a", "admin", HOSPITAL_A),
            ("nurse_b", "nurse", HOSPITAL_B), ("doctor_b", "doctor", HOSPITAL_B), ("admin_b", "admin", HOSPITAL_B),
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
        w = {"client": client, "db": conn, "people": people, "hospitals": hospitals, "patients": []}
        _populate(w)
        yield w


def call(world, who, method, path, **kwargs):
    return world["client"].request(method, path, headers=world["people"][who]["headers"], **kwargs)


def _admit(world, case, nurse, doctor, before_treatment=None):
    name = f"Patient {uuid.uuid4().hex[:6]}"
    world["patients"].append(name)
    body = call(world, nurse, "POST", "/api/patients/intake", json={"full_name": name, "gender": "Male", **case["intake"]})
    assert body.status_code == 201, body.text
    accident = body.json()["accident_record"]["accident_id"]
    run = call(world, nurse, "POST", f"/api/severity/accidents/{accident}/run")
    assert run.status_code == 200, run.text
    prediction = run.json()["prediction"]
    if before_treatment:
        prediction = before_treatment(accident)
    treated = call(world, doctor, "POST", "/api/treatments", json={"accident_id": accident, **case["treatment"]})
    assert treated.status_code == 201, treated.text
    return accident, prediction


def _doctor_prediction(world, accident, model, doctor):
    view = call(world, doctor, "GET", f"/api/doctor/accidents/{accident}")
    assert view.status_code == 200, view.text
    return view.json()["models"][model]["prediction"]


def _corrected(world, accident):
    """Intake corrected before treatment and Severity run again: two predictions, history kept."""
    edited = call(world, "nurse_b", "PATCH", f"/api/patients/intake/{accident}", json={"helmet_worn": "yes"})
    assert edited.status_code == 200, edited.text
    rerun = call(world, "nurse_b", "POST", f"/api/severity/accidents/{accident}/run")
    assert rerun.status_code == 200, rerun.text
    return rerun.json()["prediction"]


def _populate(world):
    """Two episodes at A (one decided throughout) and one at B (Severity rerun, overridden)."""
    a1, severity_a1 = _admit(world, CASE_A, "nurse_a", "doctor_a")
    _admit(world, CASE_B, "nurse_a", "doctor_a")
    b1, latest = _admit(world, CASE_B, "nurse_b", "doctor_b", before_treatment=lambda b: _corrected(world, b))

    # Episode A1: nurse accepts Severity (if the model scored), doctor accepts Stay and overrides Transfer.
    if severity_a1["prediction"] != "DEFER TO CLINICAL JUDGEMENT":
        accepted = call(world, "nurse_a", "POST", f"/api/severity/predictions/{severity_a1['prediction_id']}/decision",
                        json={"action": "accept"})
        assert accepted.status_code == 201, accepted.text
    stay = _doctor_prediction(world, a1, "stay_duration", "doctor_a")
    assert call(world, "doctor_a", "POST", f"/api/doctor/accidents/{a1}/predictions/{stay['prediction_id']}/decision",
                json={"action": "accept"}).status_code == 201
    transfer = _doctor_prediction(world, a1, "transfer", "doctor_a")
    assert transfer["prediction"] == "No transfer"            # CASE_A's real model output
    assert call(world, "doctor_a", "POST", f"/api/doctor/accidents/{a1}/predictions/{transfer['prediction_id']}/decision",
                json={"action": "override", "final_decision": "Transfer", "reason": "Needs neurosurgery"}).status_code == 201

    # Episode B1: the nurse overrides its latest Severity prediction.
    choice = "HIGH" if latest["prediction"] != "HIGH" else "BORDERLINE"
    assert call(world, "nurse_b", "POST", f"/api/severity/predictions/{latest['prediction_id']}/decision",
                json={"action": "override", "final_decision": choice}).status_code == 201

    # An operational transfer request at A, separate from any prediction.
    a2 = _admit(world, CASE_A, "nurse_a", "doctor_a")[0]
    request = call(world, "admin_a", "POST", "/api/admin/transfers",
                   json={"accident_id": a2, "to_hospital_id": world["hospitals"][HOSPITAL_B], "message": "Prepare transfer."})
    assert request.status_code == 201, request.text
    world["accidents"] = [a1, b1, a2]


def summary(world, model):
    response = call(world, "government", "GET", f"/api/government/models/{model}")
    assert response.status_code == 200, response.text
    return response.json()


def db_rows(world, query, *params):
    return world["db"].execute(query, params).fetchall()


# ── Authorization ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", DATABASE_GETS)
def test_government_can_use_every_database_backed_endpoint(world, path):
    assert call(world, "government", "GET", path).status_code == 200


@pytest.mark.parametrize("path", GOVERNMENT_GETS)
def test_artifact_endpoints_answer_government_or_report_a_missing_artifact(world, path):
    assert call(world, "government", "GET", path).status_code in (200, 503)


@pytest.mark.parametrize("who", ["nurse_a", "doctor_a", "admin_a", "nurse_b", "doctor_b", "admin_b"])
@pytest.mark.parametrize("path", GOVERNMENT_GETS)
def test_staff_are_refused_every_government_endpoint(world, who, path):
    assert call(world, who, "GET", path).status_code == 403


def _token(claims, secret=None, algorithm=None):
    return jwt.encode(claims, secret or settings.secret_key, algorithm=algorithm or settings.algorithm)


def _forged(world):
    officer, nurse = world["people"]["government"]["id"], world["people"]["nurse_a"]["id"]
    later = datetime.now(timezone.utc) + timedelta(hours=1)
    return {
        "no token": None,
        "not a JWT": "not-a-token",
        "wrong signing key": _token({"sub": officer, "role": "government", "exp": later}, secret="another-secret"),
        "expired": _token({"sub": officer, "role": "government", "exp": datetime.now(timezone.utc) - timedelta(minutes=1)}),
        "no subject": _token({"role": "government", "exp": later}),
        "unknown subject claiming government": _token({"sub": str(uuid.uuid4()), "role": "government", "exp": later}),
        "unsigned (alg none)": ".".join(base64.urlsafe_b64encode(json.dumps(part).encode()).rstrip(b"=").decode()
                                        for part in ({"alg": "none", "typ": "JWT"}, {"sub": officer, "role": "government"})) + ".",
    }, {
        # A real staff account cannot become government by claiming it in the token.
        "nurse claiming government": _token({"sub": nurse, "role": "government", "exp": later}),
    }


@pytest.mark.parametrize("path", ["/api/government/statistics", "/api/government/models/severity"])
def test_malformed_or_forged_identities_are_refused(world, path):
    invalid, staff_claims = _forged(world)
    for label, token in invalid.items():
        headers = {} if token is None else {"Authorization": f"Bearer {token}"}
        assert world["client"].get(path, headers=headers).status_code == 401, label
    for label, token in staff_claims.items():
        assert world["client"].get(path, headers={"Authorization": f"Bearer {token}"}).status_code == 403, label


def test_a_user_row_with_a_government_role_is_not_a_government_officer(world):
    """Government access comes only from a Government_Officer record, never a role string."""
    rogue = str(uuid.uuid4())
    with pytest.raises(psycopg.errors.CheckViolation):
        world["db"].execute('insert into "User" (user_id, name, email, role) values (%s, %s, %s, %s)',
                            (rogue, "Rogue", "rogue@example.org", "government"))


def test_government_needs_no_hospital(world):
    columns = [c.name for c in world["db"].execute('select * from "Government_Officer" limit 0').description]
    assert "hospital_id" not in columns
    assert call(world, "government", "GET", "/api/government/models/severity").status_code == 200


def _staff_endpoints(world):
    accident = world["accidents"][0]
    prediction = db_rows(world, 'select prediction_id::text from "Model_Prediction" where accident_id = %s limit 1', accident)[0][0]
    transfer = db_rows(world, 'select transfer_id::text from "Transfer" limit 1')[0][0]
    patient = db_rows(world, 'select patient_id::text from "Accident_Record" where accident_id = %s', accident)[0][0]
    hospital = world["hospitals"][HOSPITAL_A]
    return [
        ("GET", "/api/patients/my", None),
        ("POST", "/api/patients/intake", {"full_name": "X", **CASE_A["intake"]}),
        ("PATCH", f"/api/patients/intake/{accident}", {"age": "9"}),
        ("GET", "/api/patients/transfer-requests", None),
        ("POST", f"/api/patients/transfer-requests/{transfer}/process", {"outcome": "not_transferred", "note": "x"}),
        ("GET", f"/api/patients/{patient}", None),
        ("GET", f"/api/patients/{patient}/prognosis", None),
        ("GET", f"/api/patients/hospital/{hospital}", None),
        ("GET", f"/api/accidents/{accident}", None),
        ("GET", f"/api/accidents/{accident}/predictions", None),
        ("GET", f"/api/predictions/{prediction}", None),
        ("POST", "/api/treatments", {"accident_id": accident, **CASE_A["treatment"]}),
        ("GET", f"/api/severity/accidents/{accident}", None),
        ("POST", f"/api/severity/accidents/{accident}/run", None),
        ("POST", f"/api/severity/predictions/{prediction}/decision", {"action": "accept"}),
        ("GET", f"/api/doctor/accidents/{accident}", None),
        ("POST", f"/api/doctor/accidents/{accident}/run", None),
        ("POST", f"/api/doctor/accidents/{accident}/predictions/{prediction}/decision", {"action": "accept"}),
        ("GET", "/api/admin/patients", None),
        ("GET", f"/api/admin/episodes/{accident}", None),
        ("GET", "/api/admin/users", None),
        ("POST", "/api/admin/users", {"name": "X", "email": "x@example.org", "role": "nurse"}),
        ("GET", "/api/admin/transfers", None),
        ("POST", "/api/admin/transfers", {"accident_id": accident, "to_hospital_id": world["hospitals"][HOSPITAL_B], "message": "x"}),
        ("POST", f"/api/admin/transfers/{transfer}/authorize", {}),
        ("POST", f"/api/admin/transfers/{transfer}/reject", {"reason": "x"}),
        ("GET", "/api/admin/audit-logs", None),
        ("GET", "/api/admin/network-hospitals", None),
        ("POST", "/api/xai/explain", {"accident_id": accident, "model_key": "severity"}),
    ]


def test_government_is_refused_every_staff_patient_decision_and_transfer_endpoint(world):
    before = {table: db_rows(world, f'select count(*) from "{table}"')[0][0]
              for table in ("Patient", "Accident_Record", "Treatment", "Model_Prediction", "Prediction_Decision", "Transfer", "User")}
    transfer_status = db_rows(world, 'select transfer_id, status from "Transfer" order by transfer_id')
    for method, path, body in _staff_endpoints(world):
        response = call(world, "government", method, path, **({"json": body} if body is not None else {}))
        # Authorization runs before any body or path validation.
        assert response.status_code == 403, (method, path, response.status_code, response.text)
    after = {table: db_rows(world, f'select count(*) from "{table}"')[0][0] for table in before}
    assert after == before
    assert db_rows(world, 'select transfer_id, status from "Transfer" order by transfer_id') == transfer_status


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
@pytest.mark.parametrize("path", ["/api/government/statistics", "/api/government/models/severity",
                                  "/api/government/association-rules", "/api/government/accident-forecast"])
def test_government_endpoints_are_read_only(world, method, path):
    assert call(world, "government", method, path, json={}).status_code == 405


# ── Model statistics ─────────────────────────────────────────────────────

@pytest.mark.parametrize("model", MODELS)
def test_each_model_uses_exactly_its_contract_labels(world, model):
    data = summary(world, model)
    assert data["model_key"] == model and data["labels"] == LABELS[model]
    assert [item["label"] for item in data["current"]] == LABELS[model]
    for group in data["monthly"] + data["by_hospital"]:
        assert set(group["counts"]) == set(LABELS[model])
    text = json.dumps(data)
    for retired in RETIRED:
        assert retired not in text


def test_severity_reports_the_four_official_labels_only(world):
    data = summary(world, "severity")
    assert data["labels"] == ["HIGH", "BORDERLINE", "LOW", "DEFER TO CLINICAL JUDGEMENT"]
    # DEFER is the model's "cannot score" answer: counted as an output, never as a decision.
    assert data["decision_labels"] == ["HIGH", "BORDERLINE", "LOW"]
    assert data["unavailable"] == 0
    assert data["decision_role"] == "nurse"


def test_severity_counts_match_the_stored_history(world):
    data = summary(world, "severity")
    rows = db_rows(world, 'select accident_id::text, prediction from "Model_Prediction" where model_key = %s order by prediction_seq', "severity")
    latest: dict[str, str] = {}
    for accident, prediction in rows:
        latest[accident] = prediction
    assert data["total_predictions"] == len(rows) == 5            # 4 episodes, one rerun
    assert data["episodes_with_prediction"] == len(latest) == 4   # the rerun is not counted twice
    expected = {label: list(latest.values()).count(label) for label in LABELS["severity"]}
    assert {item["label"]: item["count"] for item in data["current"]} == expected
    assert sum(item["percentage"] or 0 for item in data["current"]) == pytest.approx(100, abs=0.2)
    decisions = data["decisions"]
    assert decisions["overridden"] == 1
    assert decisions["total"] == decisions["accepted"] + decisions["overridden"]


def latest_outputs(world, model):
    """Each episode's latest stored prediction for the model: {accident: (hospital, label)}."""
    rows = db_rows(world, 'select accident_id::text, hospital_id::text, prediction from "Model_Prediction" '
                          'where model_key = %s order by prediction_seq', model)
    return {accident: (hospital, label) for accident, hospital, label in rows}


@pytest.mark.parametrize("model", ["stay_duration", "transfer", "discharge"])
def test_doctor_model_counts_match_the_stored_history(world, model):
    data = summary(world, model)
    latest = latest_outputs(world, model)
    labels = [label for _, label in latest.values()]
    assert data["episodes_with_prediction"] == len(latest) == 4
    assert {i["label"]: i["count"] for i in data["current"]} == {label: labels.count(label) for label in LABELS[model]}
    assert {i["label"]: i["percentage"] for i in data["current"]} == {
        label: round(100 * labels.count(label) / len(labels), 1) for label in LABELS[model]}
    # Episode A1 (CASE_A, never edited) has the known real outputs.
    known = {"stay_duration": "2-3 days", "transfer": "No transfer", "discharge": "Partial recovery"}[model]
    assert latest[world["accidents"][0]][1] == known


def test_doctor_decisions_are_summarised_per_model(world):
    stay = summary(world, "stay_duration")
    assert stay["decision_role"] == "doctor"
    assert stay["decisions"] == {"total": 1, "accepted": 1, "overridden": 0, "acceptance_rate": 100.0,
                                 "distribution": [{"label": "0-1 days", "count": 0, "percentage": 0.0},
                                                  {"label": "2-3 days", "count": 1, "percentage": 100.0},
                                                  {"label": "4+ days", "count": 0, "percentage": 0.0}]}
    transfer = summary(world, "transfer")
    # The doctor overrode "No transfer" with "Transfer"; the model's own output still counts as the model said.
    assert transfer["decisions"]["overridden"] == 1 and transfer["decisions"]["acceptance_rate"] == 0.0
    assert {i["label"]: i["count"] for i in transfer["decisions"]["distribution"]} == {"Transfer": 1, "No transfer": 0}
    assert latest_outputs(world, "transfer")[world["accidents"][0]][1] == "No transfer"

    discharge = summary(world, "discharge")
    assert discharge["decisions"]["total"] == 0 and discharge["decisions"]["acceptance_rate"] is None
    assert all(item["percentage"] is None for item in discharge["decisions"]["distribution"])


def test_statistics_are_network_wide_not_hospital_scoped(world):
    hospital_a, hospital_b = world["hospitals"][HOSPITAL_A], world["hospitals"][HOSPITAL_B]
    for model in MODELS:
        data = summary(world, model)
        latest = latest_outputs(world, model)
        by_hospital = {item["hospital"]: item for item in data["by_hospital"]}
        assert set(by_hospital) == {HOSPITAL_A, HOSPITAL_B}
        assert by_hospital[HOSPITAL_A]["total"] == sum(1 for h, _ in latest.values() if h == hospital_a) == 3
        assert by_hospital[HOSPITAL_B]["total"] == sum(1 for h, _ in latest.values() if h == hospital_b) == 1
        [b_label] = [label for h, label in latest.values() if h == hospital_b]
        assert by_hospital[HOSPITAL_B]["counts"] == {label: int(label == b_label) for label in LABELS[model]}


def test_monthly_history_groups_episodes_by_admission_month(world):
    data = summary(world, "severity")
    assert len(data["monthly"]) == 12
    now = datetime.now(timezone.utc)
    window = {(now.year - (1 if now.month - i <= 0 else 0), (now.month - i - 1) % 12 + 1) for i in range(12)}
    admitted_in_window = (2026, 9) in window          # every test episode has incident_at 2026-09-01
    assert sum(m["total"] for m in data["monthly"]) == (4 if admitted_in_window else 0)
    if admitted_in_window:
        [september] = [m for m in data["monthly"] if m["month"] == "Sep 2026"]
        assert september["total"] == 4


def test_model_versions_are_reported(world):
    data = summary(world, "discharge")
    assert data["versions"] and sum(v["predictions"] for v in data["versions"]) == data["total_predictions"]


def test_responses_contain_no_patient_level_data(world):
    identifiers = set(world["patients"]) | set(world["accidents"])
    identifiers |= {r[0] for r in db_rows(world, 'select patient_id::text from "Patient"')}
    identifiers |= {r[0] for r in db_rows(world, 'select prediction_id::text from "Model_Prediction"')}
    identifiers |= {person["id"] for person in world["people"].values()}
    for path in DATABASE_GETS:
        text = call(world, "government", "GET", path).text
        for identifier in identifiers:
            assert identifier not in text, (path, identifier)
        for field in ("inputs", "full_name", "accident_id", "patient_id", "decided_by", "reason"):
            assert f'"{field}"' not in text, (path, field)


def test_operational_transfers_are_reported_separately_from_the_transfer_model(world):
    stats = call(world, "government", "GET", "/api/government/statistics").json()
    assert stats["transfers_by_status"] == {"pending_authorization": 1, "authorized": 0, "approved": 0, "rejected": 0}
    assert stats["pending_transfers"] == 1 and stats["total_transfers"] == 1
    assert "transfers_by_status" not in summary(world, "transfer")


def test_unknown_model_is_rejected(world):
    for model in ("priority", "Severity", "forecast"):
        assert call(world, "government", "GET", f"/api/government/models/{model}").status_code == 422


def test_reading_statistics_changes_nothing_and_writes_no_audit(world):
    tables = ("Model_Prediction", "Prediction_Decision", "AuditLog", "Transfer")
    before = {t: db_rows(world, f'select count(*) from "{t}"')[0][0] for t in tables}
    for path in DATABASE_GETS:
        call(world, "government", "GET", path)
    assert {t: db_rows(world, f'select count(*) from "{t}"')[0][0] for t in tables} == before
