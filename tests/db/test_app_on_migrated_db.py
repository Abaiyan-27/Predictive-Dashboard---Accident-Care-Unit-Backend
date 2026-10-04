"""The real API, started against a database built only from the migrations.

A PostgREST container serves the throwaway database and a small proxy adds the
/rest/v1 prefix the Supabase client expects, so every query below (embeds,
upserts, ordering, inserts into the history tables) runs for real. Skipped when
Docker or the PostgREST image is unavailable.
"""
from __future__ import annotations

import uuid

import pytest

from app.core.security import create_access_token
from app.services import ml_service

from .conftest import running_api

@pytest.fixture(scope="module")
def api(pg_cluster):
    db = pg_cluster.create_database("app_it")
    pg_cluster.migrate(db)
    with running_api(pg_cluster, db) as (test_client, conn):
        hospital_id = conn.execute(
            'select hospital_id from "Hospital" where name = %s', ("DGH – Vavuniya",)
        ).fetchone()[0]
        people = {}
        for role in ("nurse", "doctor"):
            user_id = uuid.uuid4()
            conn.execute('insert into "User" (user_id, name, email, role, hospital_id) '
                         "values (%s, %s, %s, %s, %s)",
                         (user_id, role, f"{role}@example.org", role, hospital_id))
            token = create_access_token({"sub": str(user_id), "role": role})
            people[role] = {"id": str(user_id), "headers": {"Authorization": f"Bearer {token}"}}
        yield {"client": test_client, "db": conn, "hospital": str(hospital_id), **people}


def predictions(conn, accident_id, model_key=None):
    query = ('select prediction_id, model_key, status, prediction, inputs, trigger_event, '
             'requested_by::text, hospital_id::text, model_version_id::text '
             'from "Model_Prediction" where accident_id = %s')
    params: list = [accident_id]
    if model_key:
        query += " and model_key = %s"
        params.append(model_key)
    return conn.execute(query + " order by prediction_seq", params).fetchall()


INTAKE = {
    "full_name": "Integration Patient",
    "age": "34",
    "gender": "Male",
    "incident_at": "2026-09-01T22:15:00+00:00",
    "mode_of_travel_during_accident": "motor cyclist",
    "collision_with": "heavy vehicle",
    "vehicle_type": "motor cyclist",
    "helmet_worn": "no",
    "passenger_type": "driver",
    "mode_of_transport": "ambulance",
    "time_to_hospital": "30 minutes - 1 hour",
    "speed": "40 - 80 km/h",
    "road_type": "bend",
    # Secondary listed first on purpose: the stored order must come from injury_order.
    "injuries": [
        {"injury_order": 2, "site_of_injury": "knee", "type_of_injury": "contusion"},
        {"injury_order": 1, "site_of_injury": "head injury", "type_of_injury": "laceration"},
    ],
}


def test_full_patient_flow_on_migrated_schema(api):
    client, conn = api["client"], api["db"]

    schema = client.get("/health/database-schema").json()
    assert schema["ready"] is True, schema

    # ── Intake ────────────────────────────────────────────────────────────
    response = client.post("/api/patients/intake", headers=api["nurse"]["headers"],
                           json={**INTAKE, "hospital_id": api["hospital"]})
    assert response.status_code == 201, response.text
    body = response.json()
    accident_id = body["accident_record"]["accident_id"]
    assert [(i["injury_order"], i["site_of_injury"]) for i in body["injuries"]] == [
        (1, "head injury"), (2, "knee")]
    # Saving intake runs no model; nurses never run the doctor models at all.
    assert "severity_prediction" not in body
    assert "stay_duration_prediction" not in body and "transfer_prediction" not in body
    assert body["severity"]["state"] == "NOT_RUN"
    assert predictions(conn, accident_id) == []

    # ── The nurse runs Severity explicitly ────────────────────────────────
    response = client.post(f"/api/severity/accidents/{accident_id}/run", headers=api["nurse"]["headers"])
    assert response.status_code == 200, response.text
    run = response.json()
    stored = predictions(conn, accident_id)
    assert {row[1] for row in stored} == {"severity"}
    severity = stored[0]
    assert severity[5] == "intake_updated"
    assert severity[6] == api["nurse"]["id"]
    assert severity[7] == api["hospital"]
    assert run["prediction"]["prediction_id"] == str(severity[0])
    assert severity[3] == run["prediction"]["prediction"]
    assert severity[4]["Site of Injury No1"] == "head injury"
    assert severity[4]["Site of injury No 2"] == "knee"
    assert set(severity[4]) == set(ml_service.model_input_fields("severity"))

    fingerprint = conn.execute('select fingerprint from "Model_Version" where model_version_id = %s',
                               (severity[8],)).fetchone()[0]
    assert fingerprint == ml_service.model_fingerprint("severity")["fingerprint"]

    injury_ids = dict(conn.execute('select injury_order, injury_id from "Injury_Description" '
                                   "where accident_id = %s", (accident_id,)).fetchall())

    # ── Edit the secondary injury: nothing swaps, ids stay, history grows ──
    edit = {"injuries": [
        {"injury_order": 1, "site_of_injury": "head injury", "type_of_injury": "laceration"},
        {"injury_order": 2, "site_of_injury": "thigh", "type_of_injury": "contusion"},
    ]}
    response = client.patch(f"/api/patients/intake/{accident_id}", headers=api["nurse"]["headers"], json=edit)
    assert response.status_code == 200, response.text
    # The earlier prediction no longer matches the inputs, so it cannot be decided.
    assert response.json()["severity"]["prediction"]["stale"] is True
    rows = conn.execute('select injury_order, injury_id, site_of_injury from "Injury_Description" '
                        "where accident_id = %s order by injury_order", (accident_id,)).fetchall()
    assert [(r[0], r[2]) for r in rows] == [(1, "head injury"), (2, "thigh")]
    assert {r[0]: r[1] for r in rows} == injury_ids
    assert len(predictions(conn, accident_id)) == 1  # saving alone stores nothing

    assert client.post(f"/api/severity/accidents/{accident_id}/run",
                       headers=api["nurse"]["headers"]).status_code == 200
    history = predictions(conn, accident_id, "severity")
    assert len(history) == 2
    assert history[0] == severity  # the first prediction is untouched
    assert history[1][4]["Site of injury No 2"] == "thigh"
    assert history[1][5] == "intake_updated"

    # Re-running on identical data does not add history rows.
    before = len(predictions(conn, accident_id))
    assert client.post(f"/api/severity/accidents/{accident_id}/run",
                       headers=api["nurse"]["headers"]).status_code == 200
    assert len(predictions(conn, accident_id)) == before

    # ── Treatment unlocks and stores all four models ───────────────────────
    response = client.post("/api/treatments", headers=api["nurse"]["headers"], json={
        "accident_id": accident_id, "hospital_id": api["hospital"], "number_of_days_stay": 2,
        "investigation_done": "x-ray", "management_name_1": "wound suturing",
        "management_name_2": "no other management",
    })
    assert response.status_code == 201, response.text
    after_treatment = [row for row in predictions(conn, accident_id) if row[5] == "treatment_recorded"]
    # Severity reads no treatment fields, so its unchanged output is not stored again.
    assert {row[1] for row in after_treatment} == {"stay_duration", "transfer", "discharge"}
    assert predictions(conn, accident_id, "severity")[-1] == history[1]
    for row in after_treatment:
        assert row[2] in {"OK", "LOW_CONFIDENCE"} and row[3] is not None, row
        assert row[4]["Investigation Done"] == "x-ray"
        assert row[6] == api["nurse"]["id"]          # the nurse who entered the treatment

    # ── Prognosis returns the stored rows and adds none ───────────────────
    patient_id = body["patient"]["patient_id"]
    count = len(predictions(conn, accident_id))
    response = client.get(f"/api/patients/{patient_id}/prognosis", headers=api["doctor"]["headers"])
    assert response.status_code == 200, response.text
    prognosis = response.json()
    assert len(predictions(conn, accident_id)) == count
    latest = {row[1]: str(row[0]) for row in predictions(conn, accident_id)}
    for key, response_key in (("severity", "severity"), ("stay_duration", "stay_duration"),
                              ("transfer", "transfer"), ("discharge", "discharge")):
        assert prognosis[response_key]["prediction_id"] == latest[key]

    # ── A human decision never changes the stored prediction ───────────────
    stay_id = latest["stay_duration"]
    model_label = prognosis["stay_duration"]["prediction"]
    override = next(label for label in ("0-1 days", "2-3 days", "4+ days") if label != model_label)
    conn.execute('insert into "Prediction_Decision" (prediction_id, model_key, final_decision, '
                 "decision_source, decided_by) values (%s, 'stay_duration', %s, 'DOCTOR', %s)",
                 (stay_id, override, api["doctor"]["id"]))
    row = conn.execute('select prediction, final_decision, decision_source, decided_by::text '
                       'from "Current_Model_Prediction" where prediction_id = %s', (stay_id,)).fetchone()
    assert row == (model_label, override, "DOCTOR", api["doctor"]["id"])

    # ── Read endpoints work against the migrated schema ────────────────────
    listing = client.get(f"/api/patients/hospital/{api['hospital']}", headers=api["doctor"]["headers"])
    assert listing.status_code == 200, listing.text
    record = next(item for item in listing.json() if item["accident_id"] == accident_id)
    assert [i["injury_order"] for i in record["Injury_Description"]] == [1, 2]
    assert client.get("/api/patients/my", headers=api["nurse"]["headers"]).status_code == 200


def test_government_officer_cannot_create_patients(api):
    client, conn = api["client"], api["db"]
    patients_before = conn.execute('select count(*) from "Patient"').fetchone()[0]
    # Government officers have no hospital and cannot reach intake at all.
    officer = uuid.uuid4()
    conn.execute('insert into "Government_Officer" (officer_id, name, email) values (%s, %s, %s)',
                 (officer, "Officer", "officer@example.org"))
    token = create_access_token({"sub": str(officer), "role": "government"})
    response = client.post("/api/patients/intake", headers={"Authorization": f"Bearer {token}"},
                           json={"full_name": "No hospital"})
    assert response.status_code == 403
    assert conn.execute('select count(*) from "Patient"').fetchone()[0] == patients_before
