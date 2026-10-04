import pytest
from fastapi import APIRouter, Depends, status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.main import app
from app.api.deps import require_nurse, require_doctor, require_admin

# Create a test router to test the dependencies
test_router = APIRouter(prefix="/test-deps", tags=["Test Deps"])

@test_router.get("/nurse-only", dependencies=[Depends(require_nurse)])
def nurse_only():
    return {"message": "Success"}

@test_router.get("/doctor-only", dependencies=[Depends(require_doctor)])
def doctor_only():
    return {"message": "Success"}

@test_router.get("/admin-only", dependencies=[Depends(require_admin)])
def admin_only():
    return {"message": "Success"}

app.include_router(test_router)

class MockDBResult:
    def __init__(self, data):
        self.data = data

@pytest.fixture
def auth_mock_supabase(mock_supabase):
    """Configures the mock_supabase to return appropriate users based on token."""
    def table_side_effect(table_name):
        mock_table = MagicMock()
        
        def execute_side_effect():
            # We determine the user to return based on how the test calls it,
            # but since we can't easily read the token from inside the mock,
            # we'll just return a flexible mock that matches what the token claims.
            pass
            
        return mock_table
        
    mock_supabase.table.side_effect = table_side_effect
    return mock_supabase


def test_no_token(client):
    response = client.get("/test-deps/nurse-only")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    # Ah, in deps.py:
    # if credentials is None:
    #    raise HTTPException(status_code=401, detail="Authentication required.")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Authentication required."


def test_invalid_token(client):
    response = client.get(
        "/test-deps/nurse-only",
        headers={"Authorization": "Bearer invalid.token.here"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Invalid or expired token."


def setup_mock_db_for_user(mock_supabase, role: str):
    """Helper to mock DB response for get_current_user."""
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            if role in ["nurse", "doctor", "admin"]:
                mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                    {"user_id": f"{role}-123", "role": role,
                     "hospital_id": "00000000-0000-0000-0000-0000000000a1"}
                ])
            else:
                mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[])
        elif table_name == "Government_Officer":
            if role == "government":
                mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                    {"officer_id": "gov-123"}
                ])
            else:
                mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[])
        return mock_table
    mock_supabase.table.side_effect = table_side_effect


def test_nurse_access(client, mock_supabase, nurse_token):
    setup_mock_db_for_user(mock_supabase, "nurse")
    
    # Nurse can access nurse-only
    response = client.get(
        "/test-deps/nurse-only",
        headers={"Authorization": f"Bearer {nurse_token}"}
    )
    assert response.status_code == status.HTTP_200_OK
    
    # Nurse CANNOT access doctor-only
    response2 = client.get(
        "/test-deps/doctor-only",
        headers={"Authorization": f"Bearer {nurse_token}"}
    )
    assert response2.status_code == status.HTTP_403_FORBIDDEN


def test_admin_access(client, mock_supabase, admin_token):
    setup_mock_db_for_user(mock_supabase, "admin")
    
    # Admin can access admin-only
    response = client.get(
        "/test-deps/admin-only",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == status.HTTP_200_OK
    
    # Intake is a nurse workflow; admins no longer pass require_nurse.
    response2 = client.get(
        "/test-deps/nurse-only",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response2.status_code == status.HTTP_403_FORBIDDEN
    
    # Admin CANNOT access doctor-only
    response3 = client.get(
        "/test-deps/doctor-only",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response3.status_code == status.HTTP_403_FORBIDDEN


def test_doctor_access(client, mock_supabase, doctor_token):
    setup_mock_db_for_user(mock_supabase, "doctor")
    
    response = client.get(
        "/test-deps/doctor-only",
        headers={"Authorization": f"Bearer {doctor_token}"}
    )
    assert response.status_code == status.HTTP_200_OK


def test_staff_without_hospital_is_refused(client, mock_supabase, admin_token):
    """The production admin with no hospital must not reach any staff endpoint."""
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "admin-123", "role": "admin", "hospital_id": None}
            ])
        return mock_table
    mock_supabase.table.side_effect = table_side_effect

    response = client.get("/test-deps/admin-only", headers={"Authorization": f"Bearer {admin_token}"})
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert "not assigned to a hospital" in response.json()["detail"]


def test_role_comes_from_database_not_token(client, mock_supabase):
    """A token claiming 'admin' for a user stored as a nurse is treated as a nurse."""
    from tests.conftest import generate_test_token
    setup_mock_db_for_user(mock_supabase, "nurse")
    forged = generate_test_token("nurse-123", "admin")
    response = client.get("/test-deps/admin-only", headers={"Authorization": f"Bearer {forged}"})
    assert response.status_code == status.HTTP_403_FORBIDDEN
