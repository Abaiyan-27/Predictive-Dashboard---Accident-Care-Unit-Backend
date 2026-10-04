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
            mock_table.select.return_value.ilike.return_value.execute.return_value = MockDBResult(data=[
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
            mock_table.select.return_value.ilike.return_value.execute.return_value = MockDBResult(data=[])
        elif table_name == "Government_Officer":
            # Found in Government_Officer table
            mock_table.select.return_value.ilike.return_value.execute.return_value = MockDBResult(data=[
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


def test_login_with_a_temporary_password_requires_a_change(client, mock_auth_supabase, mock_supabase):
    from unittest.mock import MagicMock as _Mock
    mock_auth_supabase.auth.sign_in_with_password.return_value = _Mock(
        user=_Mock(id="u-1", user_metadata={"must_change_password": True}))
    response = client.post("/api/auth/login", json={"email": "new@test.com", "password": "TempPass-generated"})
    assert response.status_code == 403
    assert response.json()["detail"] == "PASSWORD_CHANGE_REQUIRED"
    mock_supabase.table.assert_not_called()   # no profile lookup, no token


def test_change_password_verifies_the_current_one_and_clears_the_flag(client, mock_auth_supabase, mock_supabase):
    from unittest.mock import MagicMock as _Mock
    mock_auth_supabase.auth.sign_in_with_password.return_value = _Mock(
        user=_Mock(id="u-1", user_metadata={"must_change_password": True, "other": 1}))
    response = client.post("/api/auth/change-password", json={
        "email": "new@test.com", "current_password": "TempPass-generated", "new_password": "MyOwnPass123"})
    assert response.status_code == 204
    user_id, update = mock_supabase.auth.admin.update_user_by_id.call_args.args
    assert user_id == "u-1" and update["password"] == "MyOwnPass123"
    assert update["user_metadata"] == {"must_change_password": False, "other": 1}


def test_change_password_with_a_wrong_current_password_changes_nothing(client, mock_auth_supabase, mock_supabase):
    mock_auth_supabase.auth.sign_in_with_password.side_effect = Exception("invalid")
    response = client.post("/api/auth/change-password", json={
        "email": "new@test.com", "current_password": "wrong", "new_password": "MyOwnPass123"})
    assert response.status_code == 401
    mock_supabase.auth.admin.update_user_by_id.assert_not_called()


def test_login_matches_the_profile_by_the_verified_email_ignoring_case(client, mock_auth_supabase, mock_supabase):
    user = MockUser()
    user.email = "nurse_one@accidentcare.org"          # what Supabase Auth verified
    mock_auth_supabase.auth.sign_in_with_password.return_value = MockAuthResponse(user=user)
    tables = {}

    def table_side_effect(table_name):
        mock_table = tables.setdefault(table_name, MagicMock())
        if table_name == "User":
            mock_table.select.return_value.ilike.return_value.execute.return_value = MockDBResult(data=[{
                "user_id": "00000000-0000-0000-0000-000000000001", "name": "Nurse One",
                "email": "Nurse_One@AccidentCare.org", "role": "nurse", "hospital_id": None}])
        return mock_table

    mock_supabase.table.side_effect = table_side_effect
    response = client.post("/api/auth/login", json={"email": "NURSE_ONE@accidentcare.org", "password": "demo"})
    assert response.status_code == status.HTTP_200_OK
    # Looked up by the verified address, with "_" escaped so it matches only itself.
    tables["User"].select.return_value.ilike.assert_called_once_with("email", "nurse\\_one@accidentcare.org")
