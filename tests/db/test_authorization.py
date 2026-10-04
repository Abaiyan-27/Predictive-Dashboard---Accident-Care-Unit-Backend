"""Role and hospital authorization against the real API and a migrated database.

Two network hospitals (A = DGH – Vavuniya, B = DGH – Mannar), each with two or
one nurses, a doctor and an admin, plus a government officer and an admin with
no hospital (mirroring the live project). Patients are created through the API,
then every endpoint is called across roles and hospitals.
"""
from __future__ import annotations

import uuid

import pytest

from app.core.security import create_access_token

from .conftest import psycopg, running_api

HOSPITAL_A, HOSPITAL_B, HOSPITAL_C = "DGH – Vavuniya", "DGH – Mannar", "DGH – Kilinochchi"

INTAKE = {
    "age": "34",
    "incident_at": "2026-09-01T22:15:00+00:00",
    "mode_of_travel_during_accident": "motor cyclist",
    "collision_with": "heavy vehicle",
    "vehicle_type": "motor cyclist",
    "helmet_worn": "no",
    "passenger_type": "driver",
    "mode_of_transport": "ambulance",
    "time_to_hospital": "30 minutes - 1 hour",
    # Accident context: Transfer and Discharge need it to reach their completeness floors.
    "road_type": "bend", "speed": "40 - 80 km/h", "road_condition": "good", "visibility": "adequate",
    "road_signals_exist": "no", "collision_force_from": "front", "alcohol_consumption": "no",
    "illicit_drugs": "no",
    "injuries": [{"injury_order": 1, "site_of_injury": "head injury", "type_of_injury": "laceration"}],
}
TREATMENT = {
    "number_of_days_stay": 2, "investigation_done": "x-ray",
    "management_name_1": "wound suturing", "management_name_2": "no other management",
}


@pytest.fixture(scope="module")
def world(pg_cluster):
    db = pg_cluster.create_database("authz")
    pg_cluster.migrate(db)
    with running_api(pg_cluster, db) as (client, conn):
        hospitals = {
            name: str(conn.execute('select hospital_id from "Hospital" where name = %s', (name,)).fetchone()[0])
            for name in (HOSPITAL_A, HOSPITAL_B, HOSPITAL_C)
        }
        people: dict[str, dict] = {}

        def staff(key, role, hospital):
            user_id = str(uuid.uuid4())
            conn.execute('insert into "User" (user_id, name, email, role, hospital_id) values (%s, %s, %s, %s, %s)',
                         (user_id, key, f"{key}@example.org", role, hospital))
            people[key] = {"id": user_id, "hospital": hospital,
                           "headers": {"Authorization": f"Bearer {create_access_token({'sub': user_id, 'role': role})}"}}

        for key, role, hospital in [
            ("nurse_a1", "nurse", hospitals[HOSPITAL_A]), ("nurse_a2", "nurse", hospitals[HOSPITAL_A]),
            ("doctor_a", "doctor", hospitals[HOSPITAL_A]), ("admin_a", "admin", hospitals[HOSPITAL_A]),
            ("nurse_b", "nurse", hospitals[HOSPITAL_B]), ("doctor_b", "doctor", hospitals[HOSPITAL_B]),
            ("admin_b", "admin", hospitals[HOSPITAL_B]),
        ]:
            staff(key, role, hospital)

        # The live project's admin with no hospital: the constraint is NOT VALID,
        # so recreate that state exactly.
        legacy = str(uuid.uuid4())
        conn.execute('alter table "User" drop constraint user_staff_has_hospital')
        conn.execute('insert into "User" (user_id, name, email, role) values (%s, %s, %s, %s)',
                     (legacy, "Legacy admin", "legacy@example.org", "admin"))
        conn.execute('alter table "User" add constraint user_staff_has_hospital check (hospital_id is not null) not valid')
        people["legacy_admin"] = {"id": legacy, "hospital": None, "headers": {
            "Authorization": f"Bearer {create_access_token({'sub': legacy, 'role': 'admin'})}"}}

        officer = str(uuid.uuid4())
        conn.execute('insert into "Government_Officer" (officer_id, name, email) values (%s, %s, %s)',
                     (officer, "Officer", "officer@example.org"))
        people["government"] = {"id": officer, "hospital": None, "headers": {
            "Authorization": f"Bearer {create_access_token({'sub': officer, 'role': 'government'})}"}}

        def register(nurse, doctor, name):
            body = client.post("/api/patients/intake", headers=people[nurse]["headers"],
                               json={**INTAKE, "full_name": name}).json()
            accident = body["accident_record"]["accident_id"]
            # Severity is the nurse's model; the treatment (entered by the nurse) runs the doctor models.
            run = client.post(f"/api/severity/accidents/{accident}/run", headers=people[nurse]["headers"])
            assert run.status_code == 200, run.text
            response = client.post("/api/treatments", headers=people[nurse]["headers"],
                                   json={**TREATMENT, "accident_id": accident, "hospital_id": people[nurse]["hospital"]})
            assert response.status_code == 201, response.text
            predictions = dict(conn.execute(
                'select model_key, prediction_id::text from "Model_Prediction" where accident_id = %s '
                "order by prediction_seq", (accident,)).fetchall())
            return {"patient": body["patient"]["patient_id"], "accident": accident, "predictions": predictions}

        patient_x = register("nurse_a1", "doctor_a", "Patient X")   # Hospital A
        patient_y = register("nurse_b", "doctor_b", "Patient Y")    # Hospital B

        transfer = client.post("/api/admin/transfers", headers=people["admin_b"]["headers"], json={
            "accident_id": patient_y["accident"], "from_hospital_id": hospitals[HOSPITAL_B],
            "to_hospital_id": hospitals[HOSPITAL_C], "message": "Prepare Patient Y for transfer."})
        assert transfer.status_code == 201, transfer.text

        yield {"client": client, "db": conn, "people": people, "hospitals": hospitals,
               "x": patient_x, "y": patient_y, "transfer_b_to_c": transfer.json()["transfer_id"]}


def call(world, who, method, path, **kwargs):
    return world["client"].request(method, path, headers=world["people"][who]["headers"], **kwargs)


# ── Same-hospital access ──────────────────────────────────────────────────

@pytest.mark.parametrize("who", ["nurse_a1", "nurse_a2", "doctor_a", "admin_a"])
def test_same_hospital_staff_can_view_patient(world, who):
    response = call(world, who, "GET", f"/api/patients/{world['x']['patient']}")
    assert response.status_code == 200, response.text
    assert response.json()["full_name"] == "Patient X"


def test_nurse_sees_patients_registered_by_another_nurse(world):
    listed = call(world, "nurse_a2", "GET", "/api/patients/my").json()
    assert world["x"]["accident"] in {row["accident_id"] for row in listed}
    assert world["y"]["accident"] not in {row["accident_id"] for row in listed}


def test_nurse_can_edit_another_nurses_patient_in_same_hospital(world):
    response = call(world, "nurse_a2", "PATCH", f"/api/patients/intake/{world['x']['accident']}",
                    json={"helmet_worn": "no"})
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("path", [
    "/api/patients/{patient}/prognosis",
    "/api/accidents/{accident}",
    "/api/accidents/{accident}/predictions",
])
def test_doctor_can_use_own_hospital_records(world, path):
    response = call(world, "doctor_a", "GET", path.format(**world["x"]))
    assert response.status_code == 200, response.text


def test_doctor_can_explain_own_hospital_prediction(world):
    response = call(world, "doctor_a", "POST", "/api/xai/explain",
                    json={"model_name": "stay_duration", "accident_id": world["x"]["accident"]})
    assert response.status_code == 200, response.text


def test_admin_hospital_list_contains_only_own_patients(world):
    response = call(world, "admin_a", "GET", f"/api/patients/hospital/{world['hospitals'][HOSPITAL_A]}")
    accidents = {row["accident_id"] for row in response.json()}
    assert world["x"]["accident"] in accidents and world["y"]["accident"] not in accidents


# ── Cross-hospital access ─────────────────────────────────────────────────

@pytest.mark.parametrize("who", ["nurse_a1", "doctor_a", "admin_a"])
@pytest.mark.parametrize("path", [
    "/api/patients/{patient}",
    "/api/accidents/{accident}",
    "/api/accidents/{accident}/predictions",
])
def test_other_hospital_records_are_not_found(world, who, path):
    response = call(world, who, "GET", path.format(**world["y"]))
    assert response.status_code == 404, response.text


@pytest.mark.parametrize("who", ["nurse_a1", "doctor_a", "admin_a"])
def test_other_hospital_patient_list_is_forbidden(world, who):
    response = call(world, who, "GET", f"/api/patients/hospital/{world['hospitals'][HOSPITAL_B]}")
    assert response.status_code == 403


def test_nurse_cannot_edit_other_hospital_patient(world):
    response = call(world, "nurse_a1", "PATCH", f"/api/patients/intake/{world['y']['accident']}",
                    json={"full_name": "Hijacked"})
    assert response.status_code == 404
    name = world["db"].execute('select full_name from "Patient" where patient_id = %s',
                               (world["y"]["patient"],)).fetchone()[0]
    assert name == "Patient Y"


def test_doctor_cannot_get_prognosis_of_other_hospital(world):
    assert call(world, "doctor_a", "GET", f"/api/patients/{world['y']['patient']}/prognosis").status_code == 404


@pytest.mark.parametrize("hospital_key, expected", [(HOSPITAL_A, 404), (HOSPITAL_B, 403)])
def test_nurse_cannot_treat_other_hospital_patient(world, hospital_key, expected):
    before = world["db"].execute('select count(*) from "Treatment"').fetchone()[0]
    response = call(world, "nurse_a1", "POST", "/api/treatments", json={
        **TREATMENT, "accident_id": world["y"]["accident"], "hospital_id": world["hospitals"][hospital_key]})
    assert response.status_code == expected
    assert world["db"].execute('select count(*) from "Treatment"').fetchone()[0] == before


def test_doctor_cannot_explain_other_hospital_prediction(world):
    response = call(world, "doctor_a", "POST", "/api/xai/explain",
                    json={"model_name": "stay_duration", "accident_id": world["y"]["accident"]})
    assert response.status_code == 404


@pytest.mark.parametrize("model_key", ["severity", "stay_duration", "transfer", "discharge"])
def test_prediction_ids_from_other_hospital_are_not_found(world, model_key):
    prediction_id = world["y"]["predictions"][model_key]
    assert call(world, "doctor_a", "GET", f"/api/predictions/{prediction_id}").status_code == 404
    assert call(world, "doctor_b", "GET", f"/api/predictions/{prediction_id}").status_code == 200


# ── Transfers ─────────────────────────────────────────────────────────────

def test_transfers_are_listed_only_for_involved_hospitals(world):
    transfer = world["transfer_b_to_c"]
    assert transfer not in {t["transfer_id"] for t in call(world, "admin_a", "GET", "/api/admin/transfers").json()}
    assert transfer in {t["transfer_id"] for t in call(world, "admin_b", "GET", "/api/admin/transfers").json()}


def test_uninvolved_admin_cannot_change_transfer(world):
    for action, body in (("authorize", {}), ("reject", {"reason": "x"})):
        response = call(world, "admin_a", "POST", f"/api/admin/transfers/{world['transfer_b_to_c']}/{action}", json=body)
        assert response.status_code == 404
    status = world["db"].execute('select status from "Transfer" where transfer_id = %s',
                                 (world["transfer_b_to_c"],)).fetchone()[0]
    assert status == "pending_authorization"


def test_cannot_request_transfer_of_other_hospital_patient(world):
    h = world["hospitals"]
    assert call(world, "admin_a", "POST", "/api/admin/transfers", json={
        "accident_id": world["y"]["accident"], "from_hospital_id": h[HOSPITAL_A],
        "to_hospital_id": h[HOSPITAL_C], "message": "m"}).status_code == 404
    assert call(world, "admin_a", "POST", "/api/admin/transfers", json={
        "accident_id": world["x"]["accident"], "from_hospital_id": h[HOSPITAL_B],
        "to_hospital_id": h[HOSPITAL_C], "message": "m"}).status_code == 403
    # Requesting transfers is an admin operation (Phase 6), not a doctor's.
    assert call(world, "doctor_a", "POST", "/api/admin/transfers", json={
        "accident_id": world["x"]["accident"], "to_hospital_id": h[HOSPITAL_C], "message": "m"}).status_code == 403


# ── Role restrictions ─────────────────────────────────────────────────────

@pytest.mark.parametrize("who", ["nurse_a1", "admin_a"])
def test_doctor_model_endpoints_are_doctor_only(world, who):
    x = world["x"]
    assert call(world, who, "GET", f"/api/patients/{x['patient']}/prognosis").status_code == 403
    assert call(world, who, "POST", "/api/xai/explain",
                json={"model_name": "transfer", "accident_id": x["accident"]}).status_code == 403


@pytest.mark.parametrize("who", ["doctor_a", "admin_a"])
def test_only_nurses_record_treatment(world, who):
    """Treatment is a model input, and every model input is entered by a nurse."""
    x = world["x"]
    assert call(world, who, "POST", "/api/treatments", json={
        **TREATMENT, "accident_id": x["accident"], "hospital_id": world["people"][who]["hospital"]}).status_code == 403


def test_nurse_sees_only_severity_predictions(world):
    x = world["x"]
    history = call(world, "nurse_a1", "GET", f"/api/accidents/{x['accident']}/predictions").json()
    assert {row["model_key"] for row in history} == {"severity"}
    for model_key in ("stay_duration", "transfer", "discharge"):
        assert call(world, "nurse_a1", "GET", f"/api/predictions/{x['predictions'][model_key]}").status_code == 404
    detail = call(world, "nurse_a1", "GET", f"/api/accidents/{x['accident']}").json()
    assert "severity_prediction" in detail and "transfer_prediction" not in detail


@pytest.mark.parametrize("who", ["doctor_a", "admin_a"])
def test_patient_intake_is_a_nurse_workflow(world, who):
    response = call(world, who, "POST", "/api/patients/intake", json={**INTAKE, "full_name": "Z"})
    assert response.status_code == 403


def test_intake_hospital_cannot_be_chosen_by_client(world):
    before = world["db"].execute('select count(*) from "Patient"').fetchone()[0]
    response = call(world, "nurse_a1", "POST", "/api/patients/intake",
                    json={**INTAKE, "full_name": "Z", "hospital_id": world["hospitals"][HOSPITAL_B]})
    assert response.status_code == 403
    assert world["db"].execute('select count(*) from "Patient"').fetchone()[0] == before


def test_role_and_hospital_come_from_the_database(world):
    """A token claiming 'doctor' for a nurse account still acts as the nurse."""
    nurse = world["people"]["nurse_a1"]["id"]
    forged = {"Authorization": f"Bearer {create_access_token({'sub': nurse, 'role': 'doctor'})}"}
    response = world["client"].get(f"/api/patients/{world['x']['patient']}/prognosis", headers=forged)
    assert response.status_code == 403


@pytest.mark.parametrize("method, path", [
    ("GET", "/api/admin/users"), ("GET", "/api/admin/audit-logs"), ("GET", "/api/admin/hospitals"),
])
def test_admin_endpoints_are_limited_to_own_hospital(world, method, path):
    response = call(world, "admin_a", method, path)
    assert response.status_code == 200
    hospital_a = world["hospitals"][HOSPITAL_A]
    for row in response.json():
        assert row["hospital_id"] == hospital_a, row


def test_admin_cannot_view_another_hospitals_details(world):
    response = call(world, "admin_a", "GET", f"/api/admin/hospitals/{world['hospitals'][HOSPITAL_B]}")
    assert response.status_code == 403


# ── Government and hospital-less accounts ─────────────────────────────────

STAFF_ENDPOINTS = [
    ("GET", "/api/patients/my", None),
    ("GET", "/api/patients/{patient}", None),
    ("GET", "/api/accidents/{accident}", None),
    ("GET", "/api/accidents/{accident}/predictions", None),
    ("GET", "/api/patients/{patient}/prognosis", None),
    ("POST", "/api/patients/intake", {"full_name": "Z"}),
    ("PATCH", "/api/patients/intake/{accident}", {"helmet_worn": "no"}),
    ("POST", "/api/xai/explain", {"model_name": "transfer", "accident_id": "{accident}"}),
    ("GET", "/api/admin/transfers", None),
    ("GET", "/api/admin/users", None),
    ("GET", "/api/admin/audit-logs", None),
    ("GET", "/api/admin/hospitals", None),
]


@pytest.mark.parametrize("who", ["government", "legacy_admin"])
@pytest.mark.parametrize("method, path, body", STAFF_ENDPOINTS)
def test_non_hospital_accounts_cannot_use_staff_endpoints(world, who, method, path, body):
    x = world["x"]
    json_body = {k: (v.format(**x) if isinstance(v, str) else v) for k, v in (body or {}).items()} or None
    response = call(world, who, method, path.format(**x), json=json_body)
    assert response.status_code == 403, (who, path, response.text)


def test_hospital_less_admin_is_told_why(world):
    response = call(world, "legacy_admin", "GET", "/api/admin/users")
    assert "not assigned to a hospital" in response.json()["detail"]


@pytest.mark.parametrize("path", ["/api/government/accident-trends", "/api/government/association-rules"])
def test_government_endpoints_are_government_only(world, path):
    assert call(world, "government", "GET", path).status_code == 200
    for who in ("nurse_a1", "doctor_a", "admin_a"):
        assert call(world, who, "GET", path).status_code == 403


def test_malformed_ids_are_rejected(world):
    assert call(world, "doctor_a", "GET", "/api/patients/not-a-uuid").status_code == 422
    assert call(world, "doctor_a", "GET", "/api/predictions/1 or 1=1").status_code == 422


# ── Decisions (enforced in the database for every writer) ─────────────────

def decide(world, prediction_id, model_key, final, source, by):
    world["db"].execute(
        'insert into "Prediction_Decision" (prediction_id, model_key, final_decision, decision_source, decided_by) '
        "values (%s, %s, %s, %s, %s)", (prediction_id, model_key, final, source, by))


def other(model_key, prediction):
    labels = {"severity": ["HIGH", "BORDERLINE", "LOW"], "stay_duration": ["0-1 days", "2-3 days", "4+ days"],
              "transfer": ["Transfer", "No transfer"],
              "discharge": ["Complete recovery", "Partial recovery", "Further interventions"]}[model_key]
    return next(label for label in labels if label != prediction)


def current(world, prediction_id):
    return world["db"].execute('select prediction from "Model_Prediction" where prediction_id = %s',
                               (prediction_id,)).fetchone()[0]


def test_nurse_records_severity_decision_for_own_hospital(world):
    pid = world["x"]["predictions"]["severity"]
    decide(world, pid, "severity", other("severity", current(world, pid)), "NURSE",
           world["people"]["nurse_a2"]["id"])


@pytest.mark.parametrize("who, model_key, source", [
    ("nurse_a1", "stay_duration", "NURSE"),    # nurse on a doctor model
    ("doctor_a", "severity", "DOCTOR"),        # doctor modifying the nurse's severity decision
    ("admin_a", "severity", "NURSE"),          # admin cannot decide
    ("admin_a", "discharge", "DOCTOR"),
    ("nurse_b", "severity", "NURSE"),          # other hospital
    ("doctor_b", "transfer", "DOCTOR"),        # other hospital
])
def test_disallowed_decisions_are_rejected(world, who, model_key, source):
    pid = world["x"]["predictions"][model_key]
    with pytest.raises(psycopg.errors.CheckViolation):
        decide(world, pid, model_key, other(model_key, current(world, pid)), source, world["people"][who]["id"])


def test_government_cannot_record_decisions(world):
    pid = world["x"]["predictions"]["severity"]
    # Officers are not User rows: the decision trigger rejects them as an
    # unknown decider, and the foreign key would reject them as well.
    with pytest.raises((psycopg.errors.CheckViolation, psycopg.errors.ForeignKeyViolation)):
        decide(world, pid, "severity", other("severity", current(world, pid)), "NURSE",
               world["people"]["government"]["id"])
