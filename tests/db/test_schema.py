"""The repository migrations alone build the schema the application needs."""
from __future__ import annotations

import json
import uuid

import joblib
import pytest

from app.contracts.severity import SEVERITY_LABELS
from app.services import ml_service, patient_service
from app.services.schema_service import MODEL_INTEGRATION_SCHEMA

from .conftest import MIGRATIONS, psycopg

# Columns of the live Supabase project (read from its PostgREST schema on
# 2026-10-03). A fresh database must have at least these, so the migrations
# describe production rather than a parallel schema.
LIVE_COLUMNS = {
    "Hospital": ["hospital_id", "name", "type", "region", "district", "address"],
    "User": ["user_id", "name", "email", "nic", "role", "hospital_id", "created_at"],
    "Government_Officer": ["officer_id", "name", "email", "nic", "created_at"],
    "Patient": ["patient_id", "full_name", "age", "gender", "ethnicity", "occupation"],
    "Socioeconomic_Info": ["patient_id", "family_income_before", "family_income_after",
                           "family_current_status", "insurance_claim_type"],
    "Accident_Record": [
        "accident_id", "patient_id", "hospital_id", "first_hospital_id", "managed_by",
        "created_by", "vehicle_type", "helmet_worn", "road_type", "speed", "created_at",
        "status", "mode_of_transport", "time_to_hospital", "collision_with", "incident_at",
        "road_condition", "visibility", "road_signals_exist", "collision_force_from",
        "alcohol_consumption", "illicit_drugs", "mode_of_travel_during_accident",
        "passenger_type", "other_injury"],
    "Injury_Description": ["injury_id", "accident_id", "site_of_injury", "type_of_injury"],
    "Treatment": ["treatment_id", "accident_id", "hospital_id", "ward_number", "reason",
                  "number_of_days_stay", "decided_by", "logged_by", "created_at",
                  "investigation_done", "management_name_1", "management_name_2"],
    "Transfer": ["transfer_id", "accident_id", "from_hospital_id", "to_hospital_id",
                 "authorized_by", "authorization_reason", "authorized_at", "requested_by",
                 "requested_at", "approved_by", "decided_at", "status", "state_of_transfer"],
    "Hospital_Patient": ["id", "accident_id", "hospital_id", "entered_at", "exited_at"],
    "AuditLog": ["log_id", "actor_id", "role", "hospital_id", "action_type", "accident_id",
                 "timestamp"],
}

NETWORK_HOSPITALS = {
    "DGH – Vavuniya", "DGH – Kilinochchi", "Teaching Hospital – Jaffna (THJ)",
    "DGH – Mullaithivu", "Base Hospital (A) – Tellipalai",
    "Base Hospital (B) – Chavakachcheri", "Base Hospital (A) – Point Pedro", "DGH – Mannar",
}


def columns(conn, table: str) -> dict[str, str]:
    rows = conn.execute(
        "select column_name, is_nullable from information_schema.columns "
        "where table_schema = 'public' and table_name = %s", (table,)
    ).fetchall()
    return dict(rows)


def foreign_keys(conn, table: str) -> set[tuple[str, str, str]]:
    rows = conn.execute("""
        select kcu.column_name, ccu.table_name, ccu.column_name
        from information_schema.table_constraints tc
        join information_schema.key_column_usage kcu
          on tc.constraint_name = kcu.constraint_name and tc.table_schema = kcu.table_schema
        join information_schema.constraint_column_usage ccu
          on tc.constraint_name = ccu.constraint_name and tc.table_schema = ccu.table_schema
        where tc.constraint_type = 'FOREIGN KEY' and tc.table_schema = 'public'
          and tc.table_name = %s
    """, (table,)).fetchall()
    return set(rows)


# ── Fixture data helpers ─────────────────────────────────────────────────

def hospital(conn, name: str | None = None) -> str:
    if name:
        return conn.execute('select hospital_id from "Hospital" where name = %s', (name,)).fetchone()[0]
    return conn.execute(
        'insert into "Hospital" (name) values (%s) returning hospital_id', (f"H {uuid.uuid4()}",)
    ).fetchone()[0]


def user(conn, role: str, hospital_id) -> str:
    return conn.execute(
        'insert into "User" (name, email, role, hospital_id) values (%s, %s, %s, %s) returning user_id',
        (role, f"{uuid.uuid4()}@example.org", role, hospital_id),
    ).fetchone()[0]


def accident(conn, hospital_id, creator) -> str:
    patient_id = conn.execute('insert into "Patient" (full_name) values (%s) returning patient_id',
                              ("Test patient",)).fetchone()[0]
    return conn.execute(
        'insert into "Accident_Record" (patient_id, hospital_id, first_hospital_id, created_by) '
        'values (%s, %s, %s, %s) returning accident_id',
        (patient_id, hospital_id, hospital_id, creator),
    ).fetchone()[0]


def model_version(conn, model_key: str, fingerprint: str = "a" * 64) -> str:
    return conn.execute(
        'insert into "Model_Version" (model_key, fingerprint, contract_version, artifact_name, source) '
        "values (%s, %s, 'v-test', 'artifact.joblib', 'test') "
        "on conflict (model_key, fingerprint) do update set source = excluded.source "
        "returning model_version_id",
        (model_key, fingerprint),
    ).fetchone()[0]


def prediction(conn, accident_id, hospital_id, model_key, *, status="OK", label="HIGH",
               version=None, inputs=None, probability=0.42) -> str:
    version = version or model_version(conn, model_key)
    inputs = inputs or {"Age": "30-39"}
    return conn.execute(
        'insert into "Model_Prediction" (accident_id, hospital_id, model_key, model_version_id, '
        "status, prediction, probability, probabilities, completeness, inputs, inputs_hash, "
        "trigger_event) values (%s, %s, %s, %s, %s, %s, %s, %s, 0.5, %s, %s, 'intake_created') "
        "returning prediction_id",
        (accident_id, hospital_id, model_key, version, status, label, probability,
         json.dumps({label: probability} if label else {}), json.dumps(inputs), "b" * 64),
    ).fetchone()[0]


@pytest.fixture
def ward(conn):
    """A network hospital with a nurse, a doctor and one accident episode."""
    hospital_id = hospital(conn, "DGH – Vavuniya")
    nurse = user(conn, "nurse", hospital_id)
    doctor = user(conn, "doctor", hospital_id)
    return {
        "hospital": hospital_id,
        "nurse": nurse,
        "doctor": doctor,
        "accident": accident(conn, hospital_id, nurse),
    }


# ── Required columns, keys and indexes ─────────────────────────────────────

@pytest.mark.parametrize("table", sorted(LIVE_COLUMNS))
def test_fresh_database_has_every_live_column(conn, table):
    missing = set(LIVE_COLUMNS[table]) - set(columns(conn, table))
    assert not missing, f"{table} lacks {missing}"


@pytest.mark.parametrize("table", sorted(MODEL_INTEGRATION_SCHEMA))
def test_fresh_database_has_every_column_the_backend_checks(conn, table):
    missing = set(MODEL_INTEGRATION_SCHEMA[table]) - set(columns(conn, table))
    assert not missing, f"{table} lacks {missing}"


@pytest.mark.parametrize("table, column", [
    ("Accident_Record", "patient_id"), ("Accident_Record", "hospital_id"),
    ("Accident_Record", "first_hospital_id"), ("Accident_Record", "created_by"),
    ("Injury_Description", "accident_id"), ("Injury_Description", "injury_order"),
    ("Treatment", "accident_id"), ("Treatment", "hospital_id"), ("Treatment", "logged_by"),
    ("Transfer", "accident_id"), ("Transfer", "from_hospital_id"), ("Transfer", "to_hospital_id"),
    ("Transfer", "requested_by"), ("Model_Prediction", "accident_id"),
    ("Model_Prediction", "model_version_id"), ("Model_Prediction", "inputs"),
    ("Prediction_Decision", "prediction_id"), ("Prediction_Decision", "decided_by"),
    ("Prediction_Decision", "decided_at"),
])
def test_required_relationships_are_not_null(conn, table, column):
    assert columns(conn, table)[column] == "NO"


@pytest.mark.parametrize("table, expected", [
    ("Accident_Record", {("patient_id", "Patient", "patient_id"),
                         ("hospital_id", "Hospital", "hospital_id"),
                         ("first_hospital_id", "Hospital", "hospital_id"),
                         ("created_by", "User", "user_id")}),
    ("Injury_Description", {("accident_id", "Accident_Record", "accident_id")}),
    ("Treatment", {("accident_id", "Accident_Record", "accident_id"),
                   ("hospital_id", "Hospital", "hospital_id"),
                   ("logged_by", "User", "user_id"), ("decided_by", "User", "user_id")}),
    ("Transfer", {("accident_id", "Accident_Record", "accident_id"),
                  ("from_hospital_id", "Hospital", "hospital_id"),
                  ("to_hospital_id", "Hospital", "hospital_id"),
                  ("requested_by", "User", "user_id"), ("authorized_by", "User", "user_id"),
                  ("approved_by", "User", "user_id")}),
    ("User", {("hospital_id", "Hospital", "hospital_id")}),
    ("Model_Prediction", {("accident_id", "Accident_Record", "accident_id"),
                          ("hospital_id", "Hospital", "hospital_id"),
                          ("requested_by", "User", "user_id"),
                          ("model_version_id", "Model_Version", "model_version_id")}),
    ("Prediction_Decision", {("prediction_id", "Model_Prediction", "prediction_id"),
                             ("decided_by", "User", "user_id")}),
])
def test_foreign_keys_exist(conn, table, expected):
    missing = expected - foreign_keys(conn, table)
    assert not missing, f"{table} lacks foreign keys {missing}"


def test_expected_indexes_exist(conn):
    names = {row[0] for row in conn.execute(
        "select indexname from pg_indexes where schemaname = 'public'").fetchall()}
    for index in ("accident_record_hospital_created_idx", "accident_record_creator_created_idx",
                  "model_prediction_history_idx", "prediction_decision_history_idx",
                  "hospital_patient_one_open_stay", "user_email_unique", "hospital_name_unique",
                  "injury_one_per_order"):
        assert index in names, index


def test_government_officers_have_no_hospital(conn):
    assert "hospital_id" not in columns(conn, "Government_Officer")
    conn.execute('insert into "Government_Officer" (name, email) values (%s, %s)',
                 ("Officer", "officer@example.org"))


def test_staff_users_need_a_hospital(conn):
    with pytest.raises(psycopg.errors.CheckViolation):
        user(conn, "nurse", None)


def test_staff_roles_are_restricted(conn):
    with pytest.raises(psycopg.errors.CheckViolation):
        user(conn, "government", hospital(conn))


def test_staff_emails_are_unique_case_insensitively(conn):
    hospital_id = hospital(conn)
    conn.execute('insert into "User" (name, email, role, hospital_id) values '
                 "('A', 'Same@Example.org', 'nurse', %s)", (hospital_id,))
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute('insert into "User" (name, email, role, hospital_id) values '
                     "('B', 'same@example.org', 'doctor', %s)", (hospital_id,))


def test_network_hospitals_are_seeded(conn):
    rows = conn.execute('select name from "Hospital" where is_network_hospital').fetchall()
    assert {row[0] for row in rows} == NETWORK_HOSPITALS


def test_accident_requires_hospital_and_creator(conn, ward):
    patient_id = conn.execute('insert into "Patient" (full_name) values (%s) returning patient_id',
                              ("X",)).fetchone()[0]
    with pytest.raises(psycopg.errors.NotNullViolation):
        conn.execute('insert into "Accident_Record" (patient_id, first_hospital_id, created_by) '
                     "values (%s, %s, %s)", (patient_id, ward["hospital"], ward["nurse"]))


def test_transfer_must_change_hospital(conn, ward):
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute('insert into "Transfer" (accident_id, from_hospital_id, to_hospital_id, requested_by) '
                     "values (%s, %s, %s, %s)",
                     (ward["accident"], ward["hospital"], ward["hospital"], ward["doctor"]))


# ── Injury order ─────────────────────────────────────────────────────────

def test_injury_order_is_explicit_and_deterministic(conn, ward):
    # Secondary written first: physical order must not decide which is primary.
    conn.execute('insert into "Injury_Description" (accident_id, injury_order, site_of_injury) '
                 "values (%s, 2, 'knee')", (ward["accident"],))
    conn.execute('insert into "Injury_Description" (accident_id, injury_order, site_of_injury) '
                 "values (%s, 1, 'head injury')", (ward["accident"],))
    rows = conn.execute('select injury_order, site_of_injury from "Injury_Description" '
                        "where accident_id = %s order by injury_order", (ward["accident"],)).fetchall()
    assert rows == [(1, "head injury"), (2, "knee")]

    record = patient_service.build_model_record(
        patient={}, accident={"Injury_Description": [
            {"injury_order": 2, "site_of_injury": "knee"},
            {"injury_order": 1, "site_of_injury": "head injury"},
        ]})
    assert record["Site of Injury No1"] == "head injury"
    assert record["Site of injury No 2"] == "knee"


def test_secondary_without_primary_stays_secondary():
    record = patient_service.build_model_record(
        patient={}, accident={"Injury_Description": [{"injury_order": 2, "site_of_injury": "knee"}]})
    assert record["Site of Injury No1"] is None
    assert record["Site of injury No 2"] == "knee"


def test_one_injury_per_slot(conn, ward):
    conn.execute('insert into "Injury_Description" (accident_id, injury_order) values (%s, 1)',
                 (ward["accident"],))
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute('insert into "Injury_Description" (accident_id, injury_order) values (%s, 1)',
                     (ward["accident"],))


@pytest.mark.parametrize("order", [None, 0, 3])
def test_injury_order_must_be_primary_or_secondary(conn, ward, order):
    with pytest.raises((psycopg.errors.CheckViolation, psycopg.errors.NotNullViolation)):
        conn.execute('insert into "Injury_Description" (accident_id, injury_order) values (%s, %s)',
                     (ward["accident"], order))


# ── Predictions ──────────────────────────────────────────────────────────

def test_output_labels_match_the_api_contracts(conn):
    rows = conn.execute('select model_key, label, human_selectable from "Model_Output_Label" '
                        "order by model_key, sort_order").fetchall()
    labels: dict[str, list[str]] = {}
    for model_key, label, _ in rows:
        labels.setdefault(model_key, []).append(label)
    assert tuple(labels["severity"]) == SEVERITY_LABELS
    for bundle in ("stay_duration", "transfer", "discharge"):
        contract = json.loads((ml_service.LOCAL_MODEL_ROOT / bundle / "model_contract.json").read_text())
        model = joblib.load(ml_service.LOCAL_MODEL_ROOT / bundle / contract["artifact"])
        label_map = contract.get("label_map") or {}
        expected = {label_map.get(str(c), str(c)) for c in model.classes_}
        assert set(labels[bundle]) == expected, bundle
    selectable = {(m, l) for m, l, h in rows if not h}
    assert selectable == {("severity", "DEFER TO CLINICAL JUDGEMENT")}


def test_prediction_is_stored_with_version_inputs_and_time(conn, ward):
    version = model_version(conn, "severity", "c" * 64)
    prediction_id = prediction(conn, ward["accident"], ward["hospital"], "severity",
                               version=version, inputs={"Age": "30-39", "Helmet Worn": "no"})
    row = conn.execute(
        'select p.model_key, p.prediction, p.status, p.probability, p.inputs, p.predicted_at, '
        'v.fingerprint, v.contract_version from "Model_Prediction" p '
        'join "Model_Version" v using (model_version_id) where prediction_id = %s',
        (prediction_id,)).fetchone()
    assert row[:3] == ("severity", "HIGH", "OK")
    assert float(row[3]) == pytest.approx(0.42)
    assert row[4] == {"Age": "30-39", "Helmet Worn": "no"}
    assert row[5] is not None
    assert row[6:] == ("c" * 64, "v-test")


def test_repeated_predictions_are_kept_and_distinguishable(conn, ward):
    old_version = model_version(conn, "severity", "d" * 64)
    new_version = model_version(conn, "severity", "e" * 64)
    first = prediction(conn, ward["accident"], ward["hospital"], "severity",
                       label="BORDERLINE", status="LOW_CONFIDENCE", version=old_version)
    second = prediction(conn, ward["accident"], ward["hospital"], "severity",
                        label="HIGH", version=new_version, inputs={"Age": "40-49"})
    history = conn.execute(
        'select prediction_id, prediction, model_version_id, inputs from "Model_Prediction" '
        "where accident_id = %s and model_key = 'severity' order by prediction_seq",
        (ward["accident"],)).fetchall()
    assert [row[0] for row in history] == [first, second]
    assert [row[1] for row in history] == ["BORDERLINE", "HIGH"]
    assert history[0][2] != history[1][2]
    current = conn.execute('select prediction_id from "Current_Model_Prediction" '
                           "where accident_id = %s and model_key = 'severity'",
                           (ward["accident"],)).fetchone()[0]
    assert current == second


@pytest.mark.parametrize("statement", [
    'update "Model_Prediction" set prediction = \'LOW\' where prediction_id = %s',
    'delete from "Model_Prediction" where prediction_id = %s',
])
def test_predictions_are_append_only(conn, ward, statement):
    prediction_id = prediction(conn, ward["accident"], ward["hospital"], "severity")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute(statement, (prediction_id,))


@pytest.mark.parametrize("model_key, status, label", [
    ("severity", "OK", "Moderate"),                       # not a contract label
    ("severity", "OK", None),                             # severity always answers
    ("severity", "OK", "DEFER TO CLINICAL JUDGEMENT"),    # DEFER only when it cannot score
    ("severity", "REFUSED", "HIGH"),                      # a refusal is DEFER
    ("stay_duration", "OK", "HIGH"),                      # another model's label
    ("stay_duration", "REFUSED", "2-3 days"),             # refused but labelled
    ("transfer", "OK", None),                             # scored but unlabelled
])
def test_invalid_predictions_are_rejected(conn, ward, model_key, status, label):
    with pytest.raises((psycopg.errors.CheckViolation, psycopg.errors.ForeignKeyViolation)):
        prediction(conn, ward["accident"], ward["hospital"], model_key, status=status, label=label)


# ── Human decisions ──────────────────────────────────────────────────────

def decide(conn, prediction_id, model_key, final, source, by, reason=None):
    return conn.execute(
        'insert into "Prediction_Decision" (prediction_id, model_key, final_decision, '
        "decision_source, decided_by, reason) values (%s, %s, %s, %s, %s, %s) "
        "returning decision_id, decided_at",
        (prediction_id, model_key, final, source, by, reason),
    ).fetchone()


def test_nurse_override_keeps_the_model_prediction(conn, ward):
    prediction_id = prediction(conn, ward["accident"], ward["hospital"], "severity", label="HIGH")
    decision_id, decided_at = decide(conn, prediction_id, "severity", "LOW", "NURSE",
                                     ward["nurse"], "Minor abrasion only")
    stored = conn.execute(
        'select p.prediction, d.final_decision, d.decision_source, d.decided_by, d.decided_at, d.reason '
        'from "Prediction_Decision" d join "Model_Prediction" p using (prediction_id) '
        "where d.decision_id = %s", (decision_id,)).fetchone()
    assert stored == ("HIGH", "LOW", "NURSE", ward["nurse"], decided_at, "Minor abrasion only")


def test_accepting_the_model_records_source_model(conn, ward):
    prediction_id = prediction(conn, ward["accident"], ward["hospital"], "severity", label="HIGH")
    decide(conn, prediction_id, "severity", "HIGH", "MODEL", ward["nurse"])
    row = conn.execute('select final_decision, decision_source from "Current_Model_Prediction" '
                       "where prediction_id = %s", (prediction_id,)).fetchone()
    assert row == ("HIGH", "MODEL")


@pytest.mark.parametrize("model_key, label, final", [
    ("stay_duration", "2-3 days", "4+ days"),
    ("transfer", "No transfer", "Transfer"),
    ("discharge", "Partial recovery", "Complete recovery"),
])
def test_doctor_overrides_each_prognosis_model(conn, ward, model_key, label, final):
    prediction_id = prediction(conn, ward["accident"], ward["hospital"], model_key, label=label)
    decide(conn, prediction_id, model_key, final, "DOCTOR", ward["doctor"])
    assert conn.execute('select prediction from "Model_Prediction" where prediction_id = %s',
                        (prediction_id,)).fetchone()[0] == label


def test_doctor_can_decide_when_the_model_refused(conn, ward):
    prediction_id = prediction(conn, ward["accident"], ward["hospital"], "stay_duration",
                               status="REFUSED", label=None, probability=None)
    decide(conn, prediction_id, "stay_duration", "0-1 days", "DOCTOR", ward["doctor"])


@pytest.mark.parametrize("case", [
    "model_source_with_different_label",
    "override_with_same_label",
    "nurse_on_doctor_model",
    "doctor_on_severity",
    "source_role_mismatch",
    "defer_is_not_a_human_decision",
    "label_of_another_model",
    "accept_a_refusal",
])
def test_invalid_decisions_are_rejected(conn, ward, case):
    severity = prediction(conn, ward["accident"], ward["hospital"], "severity", label="HIGH")
    stay = prediction(conn, ward["accident"], ward["hospital"], "stay_duration", label="2-3 days")
    refused = prediction(conn, ward["accident"], ward["hospital"], "severity",
                         status="REFUSED", label="DEFER TO CLINICAL JUDGEMENT", probability=None)
    args = {
        "model_source_with_different_label": (severity, "severity", "LOW", "MODEL", ward["nurse"]),
        "override_with_same_label": (severity, "severity", "HIGH", "NURSE", ward["nurse"]),
        "nurse_on_doctor_model": (stay, "stay_duration", "4+ days", "NURSE", ward["nurse"]),
        "doctor_on_severity": (severity, "severity", "LOW", "DOCTOR", ward["doctor"]),
        "source_role_mismatch": (severity, "severity", "LOW", "DOCTOR", ward["nurse"]),
        "defer_is_not_a_human_decision": (severity, "severity", "DEFER TO CLINICAL JUDGEMENT",
                                          "NURSE", ward["nurse"]),
        "label_of_another_model": (stay, "stay_duration", "HIGH", "DOCTOR", ward["doctor"]),
        "accept_a_refusal": (refused, "severity", "DEFER TO CLINICAL JUDGEMENT", "MODEL",
                             ward["nurse"]),
    }[case]
    with pytest.raises((psycopg.errors.CheckViolation, psycopg.errors.ForeignKeyViolation)):
        decide(conn, *args)


def test_decisions_are_append_only_and_final(conn, ward):
    prediction_id = prediction(conn, ward["accident"], ward["hospital"], "severity", label="HIGH")
    first, _ = decide(conn, prediction_id, "severity", "LOW", "NURSE", ward["nurse"])
    # A decision is final: a second one (a "revision") is refused by the database.
    with pytest.raises(psycopg.errors.UniqueViolation), conn.transaction():
        decide(conn, prediction_id, "severity", "BORDERLINE", "NURSE", ward["nurse"])
    current = conn.execute('select decision_id from "Current_Model_Prediction" where prediction_id = %s',
                           (prediction_id,)).fetchone()[0]
    assert current == first
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute('update "Prediction_Decision" set final_decision = \'HIGH\' where decision_id = %s',
                     (first,))


# ── Migrating an existing (production-like) database ───────────────────────

LEGACY = [m for m in MIGRATIONS if m.name < "202610030001"]
UPGRADE = [m for m in MIGRATIONS if m.name >= "202610030001"]


def legacy_database(pg_cluster, name):
    pg_cluster.create_database(name)
    pg_cluster.migrate(name, LEGACY)
    connection = psycopg.connect(pg_cluster.dsn(name), autocommit=True)
    # Mirror the live project: a hospital outside the network, a system admin
    # without a hospital, an episode with two injuries and one with none.
    h = connection.execute('insert into "Hospital" (name) values (\'General Hospital Colombo\') '
                           "returning hospital_id").fetchone()[0]
    admin = connection.execute('insert into "User" (name, email, role) values '
                               "('Admin', 'admin@example.org', 'admin') returning user_id").fetchone()[0]
    nurse = connection.execute('insert into "User" (name, email, role, hospital_id) values '
                               "('Nurse', 'nurse@example.org', 'nurse', %s) returning user_id",
                               (h,)).fetchone()[0]
    accidents = []
    for _ in range(2):
        patient_id = connection.execute('insert into "Patient" (full_name) values (\'P\') '
                                        "returning patient_id").fetchone()[0]
        accidents.append(connection.execute(
            'insert into "Accident_Record" (patient_id, hospital_id, first_hospital_id, created_by) '
            "values (%s, %s, %s, %s) returning accident_id", (patient_id, h, h, nurse)).fetchone()[0])
    for site in ("head injury", "knee"):
        connection.execute('insert into "Injury_Description" (accident_id, site_of_injury) values (%s, %s)',
                           (accidents[0], site))
    return connection, {"hospital": h, "admin": admin, "accidents": accidents}


def test_upgrade_preserves_existing_data(pg_cluster):
    connection, data = legacy_database(pg_cluster, "legacy_ok")
    results = pg_cluster.migrate("legacy_ok", UPGRADE)
    notices = "".join(result.stderr for result in results)
    assert f"Injury order inferred from physical row order for accident(s): {data['accidents'][0]}" in notices

    rows = connection.execute('select injury_order, site_of_injury from "Injury_Description" '
                              "where accident_id = %s order by injury_order",
                              (data["accidents"][0],)).fetchall()
    assert rows == [(1, "head injury"), (2, "knee")]
    # The hospital-less admin survives (NOT VALID check) but cannot be re-saved without one.
    assert connection.execute('select count(*) from "User" where user_id = %s',
                              (data["admin"],)).fetchone()[0] == 1
    with pytest.raises(psycopg.errors.CheckViolation):
        connection.execute('update "User" set name = \'Admin 2\' where user_id = %s', (data["admin"],))
    assert connection.execute('select count(*) from "Hospital"').fetchone()[0] == 9
    assert connection.execute('select is_network_hospital from "Hospital" where hospital_id = %s',
                              (data["hospital"],)).fetchone()[0] is False
    connection.close()


def test_upgrade_aborts_without_changes_when_data_violates_constraints(pg_cluster):
    connection, data = legacy_database(pg_cluster, "legacy_bad")
    connection.execute('insert into "Injury_Description" (accident_id, site_of_injury) values (%s, %s)',
                       (data["accidents"][0], "third"))
    connection.execute('insert into "Treatment" (accident_id, number_of_days_stay) values (%s, -1)',
                       (data["accidents"][1],))

    integrity = pg_cluster.apply("legacy_bad", UPGRADE[0])
    assert integrity.returncode == 0
    failed = pg_cluster.apply("legacy_bad", UPGRADE[1])
    assert failed.returncode != 0
    assert "Treatment missing accident/hospital/logged_by: 1" in failed.stderr
    assert "Treatment with negative days: 1" in failed.stderr
    # Nothing from the failed migration was applied.
    assert columns(connection, "Accident_Record")["hospital_id"] == "YES"

    connection.execute('delete from "Treatment"')
    pg_cluster.apply("legacy_bad", UPGRADE[1])
    injury = pg_cluster.apply("legacy_bad", UPGRADE[2])
    assert injury.returncode != 0
    assert "1 accident(s) have more than two injuries" in injury.stderr
    assert "injury_order" not in columns(connection, "Injury_Description")
    connection.close()


# ── Decisions stay inside the hospital holding the episode ─────────────────

def test_decider_must_belong_to_the_predicting_hospital(conn, ward):
    prediction_id = prediction(conn, ward["accident"], ward["hospital"], "severity", label="HIGH")
    other_hospital = hospital(conn, "DGH – Mannar")
    outsider = user(conn, "nurse", other_hospital)
    with pytest.raises(psycopg.errors.CheckViolation):
        decide(conn, prediction_id, "severity", "LOW", "NURSE", outsider)


def test_no_decisions_on_predictions_after_the_episode_moves(conn, ward):
    prediction_id = prediction(conn, ward["accident"], ward["hospital"], "severity", label="HIGH")
    new_hospital = hospital(conn, "DGH – Mannar")
    conn.execute('update "Accident_Record" set hospital_id = %s where accident_id = %s',
                 (new_hospital, ward["accident"]))
    # Neither the old hospital nor the new one may decide on the old prediction.
    new_nurse = user(conn, "nurse", new_hospital)
    for decider in (ward["nurse"], new_nurse):
        with pytest.raises(psycopg.errors.CheckViolation), conn.transaction():
            decide(conn, prediction_id, "severity", "LOW", "NURSE", decider)


def test_audit_details_are_optional_json_objects(conn):
    actor = str(uuid.uuid4())
    insert = 'insert into "AuditLog" (actor_id, role, action_type, details) values (%s, %s, %s, %s)'
    conn.execute(insert, (actor, "nurse", "severity_prediction_overridden",
                          json.dumps({"model_prediction": "HIGH", "final_decision": "LOW"})))
    conn.execute(insert, (actor, "nurse", "patient_intake", None))
    with pytest.raises(psycopg.errors.CheckViolation):
        with conn.transaction():
            conn.execute(insert, (actor, "nurse", "x", json.dumps(["not", "an", "object"])))
