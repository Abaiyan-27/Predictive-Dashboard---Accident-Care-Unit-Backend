from unittest.mock import MagicMock
from fastapi import status

class MockUser:
    pass

class MockAuthResponse:
    def __init__(self, user=None):
        self.user = user

class MockDBResult:
    def __init__(self, data):
        self.data = data

def test_login_success_nurse(client, mock_auth_supabase, mock_supabase):
    # Setup auth mock
    mock_auth_supabase.auth.sign_in_with_password.return_value = MockAuthResponse(user=MockUser())
    
    # Setup DB mock for User table
    mock_select = MagicMock()
    mock_eq = MagicMock()
    mock_execute = MagicMock()
    
    mock_supabase.table.return_value.select.return_value = mock_select
    mock_select.eq.return_value = mock_eq
    mock_eq.execute = mock_execute
    
    # DB returns user on first execute() for "User" table
    mock_execute.return_value = MockDBResult(data=[
        {
            "user_id": "00000000-0000-0000-0000-000000000001",
            "name": "Nurse Joy",
            "email": "nurse@accidentcare.org",
            "role": "nurse",
            "hospital_id": "00000000-0000-0000-0000-000000000002",
        }
    ])
    
    # Mocking hospital lookup (second db call in auth.py)
    # To handle multiple tables, we can configure side_effect on table()
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {
                    "user_id": "00000000-0000-0000-0000-000000000001",
                    "name": "Nurse Joy",
                    "email": "nurse@accidentcare.org",
                    "role": "nurse",
                    "hospital_id": "00000000-0000-0000-0000-000000000002",
                }
            ])
        elif table_name == "Hospital":
            mock_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = MockDBResult(data=[
                {"name": "General Hospital"}
            ])
        return mock_table
        
    mock_supabase.table.side_effect = table_side_effect

    # Execute
    response = client.post(
        "/api/auth/login",
        json={"email": "nurse@accidentcare.org", "password": "demo"}
    )
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "token" in data
    assert data["user"]["name"] == "Nurse Joy"
    assert data["user"]["role"] == "nurse"
    assert data["user"]["hospital_name"] == "General Hospital"


def test_login_invalid_credentials(client, mock_auth_supabase):
    # Setup auth mock to raise Exception
    mock_auth_supabase.auth.sign_in_with_password.side_effect = Exception("Invalid login")

    # Execute
    response = client.post(
        "/api/auth/login",
        json={"email": "nurse@accidentcare.org", "password": "wrongpassword"}
    )
    
    # Assert
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Incorrect email or password."


def test_login_success_government(client, mock_auth_supabase, mock_supabase):
    mock_auth_supabase.auth.sign_in_with_password.return_value = MockAuthResponse(user=MockUser())
    
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            # Not found in User table
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[])
        elif table_name == "Government_Officer":
            # Found in Government_Officer table
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {
                    "officer_id": "00000000-0000-0000-0000-000000000003",
                    "name": "Gov Official",
                    "email": "gov@accidentcare.org",
                    "nic": "123456789V"
                }
            ])
        return mock_table
        
    mock_supabase.table.side_effect = table_side_effect

    # Execute
    response = client.post(
        "/api/auth/login",
        json={"email": "gov@accidentcare.org", "password": "demo"}
    )
    
    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["user"]["role"] == "government"
    assert data["user"]["hospital_name"] == "Department of Health"
