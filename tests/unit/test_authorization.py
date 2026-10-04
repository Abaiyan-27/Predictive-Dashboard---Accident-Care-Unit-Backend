"""The central authorization rules, without a database."""
import pytest
from fastapi import HTTPException

from app.core import authorization as authz

A = "00000000-0000-0000-0000-0000000000a1"
B = "00000000-0000-0000-0000-0000000000b2"


def staff(role, hospital=A):
    return authz.principal_from_user({"user_id": f"{role}-1", "role": role, "hospital_id": hospital, "_type": "user"})


GOVERNMENT = authz.principal_from_user({"officer_id": "gov-1", "role": "government", "_type": "government_officer"})


def status_of(fn, *args):
    try:
        fn(*args)
    except HTTPException as exc:
        return exc.status_code
    return 200


def test_identity_is_built_from_the_database_row():
    nurse = staff("nurse")
    assert (nurse.id, nurse.role, nurse.hospital_id, nurse.is_staff) == ("nurse-1", "nurse", A, True)
    assert GOVERNMENT.is_government and GOVERNMENT.hospital_id is None and not GOVERNMENT.is_staff


def test_government_role_requires_an_officer_record():
    impostor = authz.principal_from_user({"user_id": "u", "role": "government", "hospital_id": A, "_type": "user"})
    assert status_of(authz.ensure_role, impostor, ["government"]) == 403
    assert status_of(authz.ensure_role, GOVERNMENT, ["government"]) == 200
    assert status_of(authz.ensure_role, GOVERNMENT, ["nurse", "doctor", "admin"]) == 403


@pytest.mark.parametrize("principal", [staff("admin", None), GOVERNMENT])
def test_accounts_without_a_hospital_cannot_reach_hospital_data(principal):
    assert status_of(authz.ensure_assigned_hospital, principal) == 403
    with pytest.raises(HTTPException):
        authz.ensure_same_hospital(principal, A, kind="patient")


@pytest.mark.parametrize("role", ["nurse", "doctor", "admin"])
def test_records_from_other_hospitals_look_missing(role):
    authz.ensure_same_hospital(staff(role), A, kind="patient")
    with pytest.raises(HTTPException) as exc:
        authz.ensure_same_hospital(staff(role), B, kind="patient")
    assert exc.value.status_code == 404


def test_client_supplied_hospital_must_match():
    assert authz.ensure_own_hospital_id(staff("nurse"), None) == A
    assert authz.ensure_own_hospital_id(staff("nurse"), A) == A
    assert status_of(authz.ensure_own_hospital_id, staff("nurse"), B) == 403


@pytest.mark.parametrize("role, visible", [
    ("nurse", {"severity"}),
    ("doctor", {"severity", "stay_duration", "transfer", "discharge"}),
    ("admin", {"severity", "stay_duration", "transfer"}),
])
def test_model_visibility_by_role(role, visible):
    assert {m for m in authz.ALL_MODELS if authz.can_view_model(staff(role), m)} == visible
    assert not any(authz.can_view_model(GOVERNMENT, m) for m in authz.ALL_MODELS)


def test_filter_predictions_removes_hidden_models():
    payload = {"accident_id": "a", "severity_prediction": {}, "stay_duration_prediction": {},
               "transfer_prediction": {}, "discharge_prediction": {}}
    assert set(authz.filter_predictions(staff("nurse"), dict(payload))) == {"accident_id", "severity_prediction"}
    assert set(authz.filter_predictions(staff("doctor"), dict(payload))) == set(payload)


@pytest.mark.parametrize("role, model_key, allowed", [
    ("nurse", "severity", True),
    ("nurse", "stay_duration", False), ("nurse", "transfer", False), ("nurse", "discharge", False),
    ("doctor", "severity", False),
    ("doctor", "stay_duration", True), ("doctor", "transfer", True), ("doctor", "discharge", True),
    ("admin", "severity", False), ("admin", "stay_duration", False),
    ("admin", "transfer", False), ("admin", "discharge", False),
])
def test_decision_roles(role, model_key, allowed):
    prediction = {"model_key": model_key, "hospital_id": A}
    result = status_of(authz.ensure_can_decide, staff(role), prediction, {"hospital_id": A})
    assert (result == 200) is allowed


@pytest.mark.parametrize("model_key", authz.ALL_MODELS)
def test_government_never_decides(model_key):
    assert status_of(authz.ensure_can_decide, GOVERNMENT, {"model_key": model_key, "hospital_id": A},
                     {"hospital_id": A}) == 403


def test_decisions_stay_in_the_predicting_hospital():
    doctor = staff("doctor")
    # Episode held by another hospital.
    assert status_of(authz.ensure_can_decide, doctor, {"model_key": "transfer", "hospital_id": A},
                     {"hospital_id": B}) == 404
    # Prediction made at another hospital before the episode arrived here.
    assert status_of(authz.ensure_can_decide, doctor, {"model_key": "transfer", "hospital_id": B},
                     {"hospital_id": A}) == 403


def test_decision_roles_match_the_database_seed():
    migration = (authz.__file__.rsplit("/app/", 1)[0]
                 + "/supabase/migrations/202610030004_model_predictions.sql")
    text = open(migration, encoding="utf-8").read()
    for model_key, roles in authz.DECISION_ROLES.items():
        (role,) = roles
        assert f"('{model_key}'," in text and f"array['{role}']" in text.split(f"('{model_key}',", 1)[1].split("\n", 1)[0]
