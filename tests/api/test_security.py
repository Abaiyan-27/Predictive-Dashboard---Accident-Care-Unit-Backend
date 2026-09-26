import pytest
from fastapi import status
from unittest.mock import MagicMock, patch

class MockDBResult:
    def __init__(self, data):
        self.data = data

def setup_mock_db_for_user(mock_supabase, role: str, hospital_id: str = "hospital-123"):
    """Helper to mock DB response for get_current_user."""
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            if role in ["nurse", "doctor", "admin"]:
                mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                    {"user_id": f"{role}-123", "role": role, "hospital_id": hospital_id}
                ])
            else:
                mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[])
        elif table_name == "Government_Officer":
            if role == "government":
                mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                    {"officer_id": "gov-123", "role": "government"}
                ])
            else:
                mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[])
        
        # Additional mocks for specific endpoints if needed by the tests
        if table_name == "Accident_Record":
            mock_table.select.return_value.eq.return_value.order.return_value.execute.return_value = MockDBResult(data=[])
        return mock_table
        
    mock_supabase.table.side_effect = table_side_effect
    return mock_supabase

# ==============================================================================
# JWT AUTHENTICATION TESTS (SEC-001, SEC-002, SEC-003)
# ==============================================================================

def test_sec_001_missing_jwt(client):
    """SEC-001: Request without authentication -> denied (401)."""
    response = client.get("/api/patients/my")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Authentication required."


def test_sec_002_invalid_jwt(client):
    """SEC-002: Request with an invalid token -> denied (401)."""
    response = client.get(
        "/api/patients/my",
        headers={"Authorization": "Bearer not.a.real.jwt"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Invalid or expired token."


@patch("app.core.security.decode_token")
def test_sec_003_expired_jwt(mock_decode, client):
    """SEC-003: Request with an expired token -> denied (401)."""
    from jose import JWTError
    mock_decode.side_effect = JWTError("Signature has expired")
    
    response = client.get(
        "/api/patients/my",
        headers={"Authorization": "Bearer expired.token.here"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Invalid or expired token."


# ==============================================================================
# ROLE-BASED ACCESS CONTROL (RBAC) TESTS (SEC-004, SEC-005, SEC-006)
# ==============================================================================

def test_sec_004_nurse_access_admin_endpoint(client, mock_supabase, nurse_token):
    """SEC-004: Nurse attempts an admin operation -> denied (403)."""
    setup_mock_db_for_user(mock_supabase, "nurse")
    
    response = client.get(
        "/api/admin/users",
        headers={"Authorization": f"Bearer {nurse_token}"}
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert "Access denied" in response.json()["detail"]


def test_sec_005_doctor_access_government_endpoint(client, mock_supabase, doctor_token):
    """SEC-005: Doctor attempts to access restricted analytics -> denied (403)."""
    setup_mock_db_for_user(mock_supabase, "doctor")
    
    response = client.get(
        "/api/government/accident-trends",
        headers={"Authorization": f"Bearer {doctor_token}"}
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert "Access denied" in response.json()["detail"]


def test_sec_006_government_access_nurse_endpoint(client, mock_supabase, admin_token):
    """SEC-006: Government user attempting to access patient-identifiable information -> denied (403)."""
    setup_mock_db_for_user(mock_supabase, "government")
    
    # We use admin_token in the test helper since government token might not be explicitly pre-defined,
    # but the mock DB will return government user because of our setup_mock_db_for_user("government")
    response = client.post(
        "/api/patients/intake",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={} # payload doesn't matter, auth should fail first
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert "Access denied" in response.json()["detail"]


# ==============================================================================
# DATA ACCESS RESTRICTION TESTING (SEC-007)
# ==============================================================================

def test_sec_007_doctor_access_other_hospital(client, mock_supabase, doctor_token):
    """SEC-007: Doctor attempting to access another hospital's patient data -> denied (403)."""
    # Doctor belongs to hospital-123
    setup_mock_db_for_user(mock_supabase, "doctor", hospital_id="hospital-123")
    
    # Attempting to fetch patients for hospital-999
    response = client.get(
        "/api/patients/hospital/hospital-999",
        headers={"Authorization": f"Bearer {doctor_token}"}
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert response.json()["detail"] == "Doctors can only view patients assigned to their hospital."


# ==============================================================================
# AUDIT LOGGING TESTING (SEC-008)
# ==============================================================================

@patch("app.api.nurse.schema_service.require_intake_schema")
@patch("app.api.nurse.patient_service.add_patient_predictions")
@patch("app.api.nurse.audit_service.log_action")
def test_sec_008_audit_trail_intake(
    mock_log_action, 
    mock_add_predictions, 
    mock_require_schema, 
    client, 
    mock_supabase, 
    mock_intake_payload, 
    nurse_token
):
    """SEC-008: Verify that applicable security-related activities are logged (e.g. Data access/creation)."""
    setup_mock_db_for_user(mock_supabase, "nurse", hospital_id="hosp-123")
    
    # Mocks for the intake to succeed
    mock_patient_table = MagicMock()
    mock_patient_table.insert.return_value.execute.return_value = MockDBResult([{"patient_id": "p-1"}])
    
    mock_accident_table = MagicMock()
    mock_accident_table.insert.return_value.execute.return_value = MockDBResult([{"accident_id": "a-1"}])
    
    mock_injury_table = MagicMock()
    mock_injury_table.insert.return_value.execute.return_value = MockDBResult([{"injury_id": "i-1"}])
    
    mock_hosp_patient_table = MagicMock()
    mock_hosp_patient_table.insert.return_value.execute.return_value = MockDBResult([{"id": 1}])
    
    def table_side_effect(table_name):
        if table_name == "User":
            mock_table = MagicMock()
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "nurse-123", "role": "nurse", "hospital_id": "hosp-123"}
            ])
            return mock_table
        elif table_name == "Patient": return mock_patient_table
        elif table_name == "Accident_Record": return mock_accident_table
        elif table_name == "Injury_Description": return mock_injury_table
        elif table_name == "Hospital_Patient": return mock_hosp_patient_table
        return MagicMock()
        
    mock_supabase.table.side_effect = table_side_effect
    
    mock_add_predictions.return_value = {
        "severity_prediction": "critical",
        "stay_duration_prediction": 10,
        "transfer_prediction": "icu",
    }
    
    response = client.post(
        "/api/patients/intake",
        headers={"Authorization": f"Bearer {nurse_token}"},
        json=mock_intake_payload
    )
    
    assert response.status_code == status.HTTP_201_CREATED
    
    # Verify Audit log was called correctly
    import uuid
    mock_log_action.assert_called_once_with(
        mock_supabase,
        actor_id="nurse-123",
        role="nurse",
        action_type="patient_intake",
        hospital_id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
        accident_id="a-1"
    )
