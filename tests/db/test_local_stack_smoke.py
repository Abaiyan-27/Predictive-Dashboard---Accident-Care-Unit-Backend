"""Pre-deployment smoke test: the real backend process on a freshly migrated database.

`uvicorn app.main:app` runs in its own process with a generated SECRET_KEY,
in front of PostgREST and the local Auth stand-in (tests/local_stack). Every
request goes over HTTP, as in deployment. Covers sign-in and session rules,
one complete workflow per role, the health endpoints, and the server log.
Skipped without Docker and the PostgREST image. Never touches Supabase.
"""
from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from jose import jwt

from app.contracts.severity import SEVERITY_BORDERLINE, SEVERITY_HIGH, SEVERITY_LOW

from .conftest import docker_ready
from .test_doctor_workflow import CASE_A

pytestmark = pytest.mark.skipif(not docker_ready(), reason="Docker with the PostgREST image is not available")

ROLES = ("nurse", "doctor", "admin", "government")
TRANSFERRED = {"outcome": "transferred", "state_of_transfer": "cervical collar used during transfer"}


@pytest.fixture(scope="module")
def stack(pg_cluster, tmp_path_factory):
    from tests.local_stack.stack import local_stack

    db = pg_cluster.create_database("local_smoke")
    pg_cluster.migrate(db)
    pg_cluster.migrate(db)          # the full chain applied twice, as in the deployment preflight
    # Generated per run: no credential is written in this file.
    accounts = {role: {"email": f"{role}.{uuid.uuid4().hex[:6]}@example.org", "password": secrets.token_urlsafe(16)}
                for role in ROLES}
    log_path = tmp_path_factory.mktemp("backend") / "backend.log"
    with local_stack(pg_cluster, db, accounts, log_path=log_path) as running:
        with httpx.Client(base_url=running["api"], timeout=120) as client:
            running["client"] = client
            running["issued"] = []
            yield running


def login(stack, role=None, *, email=None, password=None):
    account = stack["people"][role] if role else {"email": email, "password": password}
    return stack["client"].post("/api/auth/login", json={"email": account["email"], "password": account["password"]})


def token(stack, role):
    response = login(stack, role)
    assert response.status_code == 200, response.text
    value = response.json()["token"]
    stack["issued"].append(value)
    return {"Authorization": f"Bearer {value}"}


def call(stack, role_or_headers, method, path, **kwargs):
    headers = token(stack, role_or_headers) if isinstance(role_or_headers, str) else role_or_headers
    return stack["client"].request(method, path, headers=headers, **kwargs)


# ── Health ─────────────────────────────────────────────────────────────────

def test_health_endpoints_report_readiness_without_internals(stack):
    client = stack["client"]
    assert client.get("/health").json()["status"] == "ok"
    models = client.get("/health/models")
    assert models.status_code == 200
    assert all(state["loadable"] for state in models.json().values()), models.text
    schema = client.get("/health/database-schema").json()
    assert schema["ready"] is True, schema
    for body in (models.text, client.get("/health/database-schema").text):
        assert "/home/" not in body and "/tmp/" not in body and "Traceback" not in body
        assert stack["service_key"] not in body and stack["secret_key"] not in body
    # Development mode serves the docs; production mode does not (unit-tested).
    assert client.get("/docs").status_code == 200


# ── Authentication ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("role", ROLES)
def test_valid_login_for_each_role(stack, role):
    response = login(stack, role)
    assert response.status_code == 200, response.text
    user = response.json()["user"]
    assert user["role"] == role and user["email"] == stack["people"][role]["email"]
    if role == "government":
        assert user["hospital_id"] is None
    else:
        assert user["hospital_id"] == stack["hospitals"]["DGH – Vavuniya"]


def test_invalid_logins_are_refused_alike(stack):
    nurse = stack["people"]["nurse"]
    wrong_password = login(stack, email=nurse["email"], password="not-the-password")
    unknown = login(stack, email="nobody@example.org", password="whatever-password")
    assert wrong_password.status_code == unknown.status_code == 401
    assert wrong_password.json() == unknown.json() == {"detail": "Incorrect email or password."}


def test_invalid_expired_and_forged_tokens_are_refused(stack):
    nurse_id = stack["people"]["nurse"]["id"]
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    tokens = {
        "garbage": "not-a-jwt",
        "expired": jwt.encode({"sub": nurse_id, "role": "nurse", "exp": past}, stack["secret_key"], algorithm="HS256"),
        "wrong key": jwt.encode({"sub": nurse_id, "role": "nurse"}, "some-other-key-" + "x" * 32, algorithm="HS256"),
        "unsigned": jwt.encode({"sub": nurse_id, "role": "nurse"}, "", algorithm="HS256").rsplit(".", 1)[0] + ".",
        "unknown user": jwt.encode({"sub": str(uuid.uuid4()), "role": "admin"}, stack["secret_key"], algorithm="HS256"),
    }
    for name, value in tokens.items():
        response = stack["client"].get("/api/patients/my", headers={"Authorization": f"Bearer {value}"})
        assert response.status_code == 401, name
    assert stack["client"].get("/api/patients/my").status_code == 401


@pytest.mark.parametrize("role,method,path", [
    ("nurse", "GET", "/api/admin/users"), ("nurse", "GET", "/api/government/statistics"),
    ("doctor", "POST", "/api/patients/intake"), ("doctor", "GET", "/api/government/statistics"),
    ("admin", "POST", "/api/treatments"), ("admin", "GET", "/api/government/statistics"),
    ("government", "POST", "/api/patients/intake"), ("government", "GET", "/api/patients/my"),
    ("government", "GET", "/api/admin/users"),
])
def test_endpoints_outside_a_role_stay_protected(stack, role, method, path):
    assert call(stack, role, method, path, json={}).status_code == 403


def test_temporary_accounts_must_change_their_password(stack):
    email = f"new.nurse.{uuid.uuid4().hex[:6]}@example.org"
    created = call(stack, "admin", "POST", "/api/admin/users", json={"name": "New Nurse", "email": email, "role": "nurse"})
    assert created.status_code == 201, created.text
    temporary = created.json()["temporary_password"]
    assert temporary and created.json()["must_change_password"] is True

    first = login(stack, email=email, password=temporary)
    assert first.status_code == 403 and first.json()["detail"] == "PASSWORD_CHANGE_REQUIRED"
    new_password = secrets.token_urlsafe(16)
    changed = stack["client"].post("/api/auth/change-password",
                                   json={"email": email, "current_password": temporary, "new_password": new_password})
    assert changed.status_code == 204, changed.text
    assert login(stack, email=email, password=temporary).status_code == 401
    signed_in = login(stack, email=email, password=new_password)
    assert signed_in.status_code == 200 and signed_in.json()["user"]["role"] == "nurse"


def test_role_and_hospital_changes_apply_on_the_next_request(stack):
    db, nurse = stack["db"], stack["people"]["nurse"]
    headers = token(stack, "nurse")
    assert call(stack, headers, "GET", "/api/patients/transfer-requests").status_code == 200
    try:
        db.execute('update "User" set role = %s where user_id = %s', ("doctor", nurse["id"]))
        assert call(stack, headers, "GET", "/api/patients/transfer-requests").status_code == 403
        assert call(stack, headers, "GET", "/api/admin/transfers").status_code == 200   # doctor access, same token
    finally:
        db.execute('update "User" set role = %s where user_id = %s', ("nurse", nurse["id"]))
    assert call(stack, headers, "GET", "/api/patients/transfer-requests").status_code == 200

    intake = call(stack, headers, "POST", "/api/patients/intake",
                  json={"full_name": "Scope Check", "gender": "Female", "age": "40"})
    assert intake.status_code == 201, intake.text
    accident = intake.json()["accident_record"]["accident_id"]
    try:
        db.execute('update "User" set hospital_id = %s where user_id = %s', (stack["hospitals"]["DGH – Kilinochchi"], nurse["id"]))
        assert call(stack, headers, "GET", f"/api/severity/accidents/{accident}").status_code in (403, 404)
        listed = call(stack, headers, "GET", "/api/patients/my").json()
        assert accident not in {row.get("accident_id") for row in listed}
    finally:
        db.execute('update "User" set hospital_id = %s where user_id = %s', (stack["hospitals"]["DGH – Vavuniya"], nurse["id"]))
    assert call(stack, headers, "GET", f"/api/severity/accidents/{accident}").status_code == 200


# ── One complete workflow per role ─────────────────────────────────────────

def test_complete_workflow_for_every_role(stack):
    hospital_a, hospital_b = stack["hospitals"]["DGH – Vavuniya"], stack["hospitals"]["DGH – Kilinochchi"]
    nurse, doctor, admin, government = (token(stack, role) for role in ROLES)

    # Nurse: lists → intake → Severity → model prediction → accept/override.
    assert call(stack, nurse, "GET", f"/api/patients/hospital/{hospital_a}").status_code == 200
    assert call(stack, nurse, "GET", "/api/patients/my").status_code == 200
    intake = call(stack, nurse, "POST", "/api/patients/intake",
                  json={"full_name": "Smoke Patient", "gender": "Male", **CASE_A["intake"]})
    assert intake.status_code == 201, intake.text
    accident = intake.json()["accident_record"]["accident_id"]
    run = call(stack, nurse, "POST", f"/api/severity/accidents/{accident}/run")
    assert run.status_code == 200, run.text
    severity = run.json()
    prediction = severity["prediction"]
    assert prediction["prediction_id"] and prediction["status"], prediction
    version = stack["db"].execute('select v.fingerprint from "Model_Prediction" p join "Model_Version" v '
                                  'using (model_version_id) where p.prediction_id = %s', (prediction["prediction_id"],)).fetchone()
    assert version and len(version[0]) == 64          # stored with the exact model fingerprint
    if severity["can_accept"]:
        decision = {"action": "accept"}
    else:
        choice = next(label for label in (SEVERITY_HIGH, SEVERITY_BORDERLINE, SEVERITY_LOW) if label != prediction["prediction"])
        decision = {"action": "override", "final_decision": choice, "reason": "Smoke test"}
    decided = call(stack, nurse, "POST", f"/api/severity/predictions/{prediction['prediction_id']}/decision", json=decision)
    assert decided.status_code == 201, decided.text
    again = call(stack, nurse, "POST", f"/api/severity/predictions/{prediction['prediction_id']}/decision", json=decision)
    assert again.status_code == 409                     # decisions are final

    # Doctor: list → patient → Severity read-only → treatment → Stay/Transfer/Discharge → accept/override.
    listed = call(stack, doctor, "GET", f"/api/patients/hospital/{hospital_a}")
    assert accident in {row["accident_id"] for row in listed.json()}
    episode = call(stack, doctor, "GET", f"/api/doctor/accidents/{accident}").json()
    assert episode["severity"]["final_severity"] and not episode["severity"]["can_override"]
    assert call(stack, doctor, "POST", f"/api/severity/accidents/{accident}/run").status_code == 403
    treated = call(stack, doctor, "POST", "/api/treatments", json={"accident_id": accident, **CASE_A["treatment"]})
    assert treated.status_code == 201, treated.text
    models = call(stack, doctor, "GET", f"/api/doctor/accidents/{accident}").json()["models"]
    for model_key in ("stay_duration", "transfer", "discharge"):
        model = models[model_key]
        assert model["prediction"] and model["prediction"]["prediction_id"], (model_key, model)
        if model_key == "transfer":
            body = {"action": "override", "final_decision": next(l for l in model["labels"] if l != model["prediction"]["prediction"]),
                    "reason": "Smoke test"}
        else:
            body = {"action": "accept"}
        response = call(stack, doctor, "POST",
                        f"/api/doctor/accidents/{accident}/predictions/{model['prediction']['prediction_id']}/decision", json=body)
        assert response.status_code == 201, (model_key, response.text)
    explained = call(stack, doctor, "POST", "/api/xai/explain",
                     json={"accident_id": accident, "model_name": "discharge"})
    assert explained.status_code in (200, 422), explained.text

    # Admin: patients with Stay/Transfer → transfer request → authorize → audit log.
    patients = call(stack, admin, "GET", "/api/admin/patients")
    assert patients.status_code == 200 and int(patients.headers["X-Total-Count"]) >= 1
    assert call(stack, admin, "GET", f"/api/admin/episodes/{accident}").status_code == 200
    requested = call(stack, admin, "POST", "/api/admin/transfers",
                     json={"accident_id": accident, "to_hospital_id": hospital_b, "message": "Smoke test transfer"})
    assert requested.status_code == 201, requested.text
    transfer_id = requested.json()["transfer_id"]
    authorized = call(stack, admin, "POST", f"/api/admin/transfers/{transfer_id}/authorize", json={"note": "Bed confirmed"})
    assert authorized.status_code == 200, authorized.text

    # Nurse: the authorized transfer → process it.
    pending = {t["transfer_id"]: t for t in call(stack, nurse, "GET", "/api/patients/transfer-requests").json()}
    assert pending[transfer_id]["allowed_actions"] == ["complete", "not_transferred"]
    processed = call(stack, nurse, "POST", f"/api/patients/transfer-requests/{transfer_id}/process", json=TRANSFERRED)
    assert processed.status_code == 200, processed.text

    actions = {entry["action_type"] for entry in call(stack, admin, "GET", "/api/admin/audit-logs").json()}
    assert {"patient_registered", "transfer_requested", "transfer_authorized", "transfer_completed"} <= actions or \
           {"transfer_requested", "transfer_authorized", "transfer_completed"} <= actions, actions

    # A second episode: the admin rejects the request instead.
    second = call(stack, nurse, "POST", "/api/patients/intake", json={"full_name": "Smoke Two", "gender": "Female", "age": "50"})
    second_id = second.json()["accident_record"]["accident_id"]
    request_two = call(stack, admin, "POST", "/api/admin/transfers",
                       json={"accident_id": second_id, "to_hospital_id": hospital_b, "message": "Second request"})
    assert request_two.status_code == 201, request_two.text
    rejected = call(stack, admin, "POST", f"/api/admin/transfers/{request_two.json()['transfer_id']}/reject",
                    json={"reason": "Not needed"})
    assert rejected.status_code == 200, rejected.text

    # Government: overview → each model page → forecast → association rules.
    statistics = call(stack, government, "GET", "/api/government/statistics")
    assert statistics.status_code == 200, statistics.text
    for model_key in ("severity", "stay_duration", "transfer", "discharge"):
        page = call(stack, government, "GET", f"/api/government/models/{model_key}")
        assert page.status_code == 200, (model_key, page.text)
    forecast = call(stack, government, "GET", "/api/government/accident-forecast", params={"horizon": 6})
    assert forecast.status_code == 200 and forecast.json()["status"] == "OK", forecast.text
    assert call(stack, government, "GET", "/api/government/accident-historical-trends").status_code == 200
    rules = call(stack, government, "GET", "/api/government/association-rules")
    assert rules.status_code == 200, rules.text


# ── The server log (run last in this module) ───────────────────────────────

def test_zz_server_log_holds_no_secrets_tokens_or_passwords(stack):
    text = stack["log_path"].read_text(encoding="utf-8")
    assert "Application startup complete" in text and "Model startup" in text
    forbidden = [stack["secret_key"], stack["service_key"], *stack["issued"],
                 *(person["password"] for person in stack["people"].values())]
    assert all(value not in text for value in forbidden)
    assert "Bearer " not in text and "authorization" not in text.lower()
