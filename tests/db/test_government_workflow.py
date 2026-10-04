"""Government access: forecasting and association-rule mining only, read-only,
against the real API and a migrated database (PostgREST), with the real models.

Patient models, decisions and operational statistics are not available to
government officers; those endpoints do not exist.

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
from .test_doctor_workflow import CASE_A, CASE_B

HOSPITAL_A, HOSPITAL_B = "DGH – Vavuniya", "DGH – Mannar"
# Every Government endpoint, as the frontend calls them: forecasting and data mining.
GOVERNMENT_GETS = [
    "/api/government/accident-trends",
    "/api/government/accident-historical-trends",
    "/api/government/accident-forecast?horizon=6",
    "/api/government/association-rules?limit=5",
]
# Endpoints backed by the database (the others read installed artifacts and
# may answer 503 where an artifact is not installed).
DATABASE_GETS = ["/api/government/accident-trends"]
# Patient-model and operational statistics: removed, for every caller.
REMOVED_GETS = ["/api/government/statistics",
                *(f"/api/government/models/{model}" for model in ("severity", "stay_duration", "transfer", "discharge"))]


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
    treated = call(world, nurse, "POST", "/api/treatments", json={"accident_id": accident, **case["treatment"]})
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


@pytest.mark.parametrize("path", ["/api/government/accident-trends", "/api/government/association-rules"])
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
    assert call(world, "government", "GET", "/api/government/accident-trends").status_code == 200


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
@pytest.mark.parametrize("path", ["/api/government/accident-trends", "/api/government/accident-historical-trends",
                                  "/api/government/association-rules", "/api/government/accident-forecast"])
def test_government_endpoints_are_read_only(world, method, path):
    assert call(world, "government", method, path, json={}).status_code == 405


# ── Only forecasting and data mining ──────────────────────────────────────

@pytest.mark.parametrize("who", ["government", "nurse_a", "doctor_a", "admin_a"])
@pytest.mark.parametrize("path", REMOVED_GETS)
def test_patient_model_and_operational_statistics_are_not_served(world, who, path):
    assert call(world, who, "GET", path).status_code == 404


def test_government_routes_are_only_forecasting_and_association_rules():
    from fastapi.routing import APIRoute

    from app.main import app

    paths = sorted(r.path for r in app.routes if isinstance(r, APIRoute) and r.path.startswith("/api/government"))
    assert paths == ["/api/government/accident-forecast", "/api/government/accident-historical-trends",
                     "/api/government/accident-trends", "/api/government/association-rules"]


def test_accident_trends_are_monthly_counts_only(world):
    trends = call(world, "government", "GET", "/api/government/accident-trends").json()
    assert len(trends) == 12 and all(set(item) == {"month", "count"} for item in trends)
    assert sum(item["count"] for item in trends) == db_rows(world, 'select count(*) from "Accident_Record"')[0][0]


def test_responses_contain_no_patient_level_data(world):
    identifiers = set(world["patients"]) | set(world["accidents"])
    identifiers |= {r[0] for r in db_rows(world, 'select patient_id::text from "Patient"')}
    identifiers |= {r[0] for r in db_rows(world, 'select prediction_id::text from "Model_Prediction"')}
    identifiers |= {person["id"] for person in world["people"].values()}
    for path in GOVERNMENT_GETS:
        text = call(world, "government", "GET", path).text
        for identifier in identifiers:
            assert identifier not in text, (path, identifier)
        for field in ("inputs", "full_name", "accident_id", "patient_id", "decided_by", "reason"):
            assert f'"{field}"' not in text, (path, field)


def test_reading_government_data_changes_nothing_and_writes_no_audit(world):
    tables = ("Model_Prediction", "Prediction_Decision", "AuditLog", "Transfer")
    before = {t: db_rows(world, f'select count(*) from "{t}"')[0][0] for t in tables}
    for path in GOVERNMENT_GETS:
        call(world, "government", "GET", path)
    assert {t: db_rows(world, f'select count(*) from "{t}"')[0][0] for t in tables} == before
