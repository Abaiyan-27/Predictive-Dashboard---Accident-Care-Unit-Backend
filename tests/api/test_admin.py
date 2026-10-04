import pytest
from unittest.mock import MagicMock, patch
from fastapi import status

class MockDBResult:
    def __init__(self, data):
        self.data = data


ADMIN_HOSPITAL = "00000000-0000-0000-0000-000000000001"

def test_get_hospitals(client, mock_supabase, admin_token):
    mock_supabase.table.return_value.select.return_value.execute.return_value = MockDBResult(data=[
        {"hospital_id": "00000000-0000-0000-0000-000000000001", "name": "General Hospital"}
    ])
    
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "00000000-0000-0000-0000-000000000002", "role": "admin", "hospital_id": ADMIN_HOSPITAL}
            ])
        elif table_name == "Hospital":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"hospital_id": ADMIN_HOSPITAL, "name": "General Hospital"}
            ])
            hospital_tables.append(mock_table)
        return mock_table

    hospital_tables = []
    mock_supabase.table.side_effect = table_side_effect

    response = client.get(
        "/api/admin/hospitals",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()) == 1
    # Only the admin's own hospital is queried.
    hospital_tables[0].select.return_value.eq.assert_called_once_with("hospital_id", ADMIN_HOSPITAL)


def test_admin_cannot_read_another_hospital(client, mock_supabase, admin_token):
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "00000000-0000-0000-0000-000000000002", "role": "admin", "hospital_id": ADMIN_HOSPITAL}
            ])
        return mock_table
    mock_supabase.table.side_effect = table_side_effect

    response = client.get(
        "/api/admin/hospitals/00000000-0000-0000-0000-0000000000b2",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_create_hospital_is_not_available(client, mock_supabase, admin_token):
    """Hospital admins cannot add hospitals; the network is managed by migrations."""
    mock_hosp_table = MagicMock()
    mock_hosp_table.insert.return_value.execute.return_value = MockDBResult(data=[
        {"hospital_id": "00000000-0000-0000-0000-000000000003", "name": "New Hospital"}
    ])
    
    mock_audit_table = MagicMock()
    mock_audit_table.insert.return_value.execute.return_value = MockDBResult(data=[{"id": 1}])
    
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "00000000-0000-0000-0000-000000000002", "role": "admin", "hospital_id": ADMIN_HOSPITAL}
            ])
            return mock_table
        elif table_name == "Hospital":
            return mock_hosp_table
        elif table_name == "AuditLog":
            return mock_audit_table
        return MagicMock()
        
    mock_supabase.table.side_effect = table_side_effect

    response = client.post(
        "/api/admin/hospitals",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "name": "New Hospital",
            "region": "North",
            "type": "General",
            "capacity": 500,
            "tier": "Tier 1"
        }
    )
    
    assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    mock_hosp_table.insert.assert_not_called()


def test_create_user_success(client, mock_supabase, admin_token):
    mock_auth_result = MagicMock()
    mock_auth_result.user.id = "00000000-0000-0000-0000-000000000004"
    mock_supabase.auth.admin.create_user.return_value = mock_auth_result
    
    mock_user_table = MagicMock()
    mock_user_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(
        data=[{"user_id": "00000000-0000-0000-0000-000000000002", "role": "admin", "hospital_id": ADMIN_HOSPITAL}]
    )  # deps.py check
    mock_user_table.select.return_value.ilike.return_value.execute.return_value = MockDBResult(data=[])
    # Insert profile
    mock_user_table.insert.return_value.execute.return_value = MockDBResult(data=[
        {"user_id": "00000000-0000-0000-0000-000000000004", "email": "doctor@test.com", "name": "Dr. Test", "role": "doctor"}
    ])
    
    mock_audit_table = MagicMock()
    mock_audit_table.insert.return_value.execute.return_value = MockDBResult(data=[{"id": 1}])
    
    def table_side_effect(table_name):
        if table_name == "User":
            return mock_user_table
        elif table_name == "AuditLog":
            return mock_audit_table
        elif table_name == "Government_Officer":
            officer_table = MagicMock()
            officer_table.select.return_value.ilike.return_value.execute.return_value = MockDBResult(data=[])
            return officer_table
        return MagicMock()
        
    mock_supabase.table.side_effect = table_side_effect
    
    response = client.post(
        "/api/admin/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "email": "doctor@test.com",
            "password": "SecurePassword123",
            "name": "Dr. Test",
            "role": "doctor",
            "hospital_id": "00000000-0000-0000-0000-000000000001"
        }
    )
    
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["user_id"] == "00000000-0000-0000-0000-000000000004"
    mock_supabase.auth.admin.create_user.assert_called_once()
    mock_audit_table.insert.assert_called_once()
    assert mock_user_table.insert.call_args.args[0]["hospital_id"] == ADMIN_HOSPITAL


def test_create_user_rollback_on_profile_fail(client, mock_supabase, admin_token):
    mock_auth_result = MagicMock()
    mock_auth_result.user.id = "00000000-0000-0000-0000-000000000004"
    mock_supabase.auth.admin.create_user.return_value = mock_auth_result
    
    mock_user_table = MagicMock()
    mock_user_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(
        data=[{"user_id": "00000000-0000-0000-0000-000000000002", "role": "admin", "hospital_id": ADMIN_HOSPITAL}]
    )
    mock_user_table.select.return_value.ilike.return_value.execute.return_value = MockDBResult(data=[])
    # Simulate DB failure on profile insert
    mock_user_table.insert.return_value.execute.side_effect = Exception("DB Fail")
    
    def table_side_effect(table_name):
        if table_name == "User":
            return mock_user_table
        if table_name == "Government_Officer":
            officer_table = MagicMock()
            officer_table.select.return_value.ilike.return_value.execute.return_value = MockDBResult(data=[])
            return officer_table
        return MagicMock()
        
    mock_supabase.table.side_effect = table_side_effect
    
    response = client.post(
        "/api/admin/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "email": "doctor@test.com",
            "password": "SecurePassword123",
            "name": "Dr. Test",
            "role": "doctor",
            "hospital_id": "00000000-0000-0000-0000-000000000001"
        }
    )
    
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to create user profile" in response.json()["detail"]
    
    # Assert that delete_user was called to rollback the auth account
    mock_supabase.auth.admin.delete_user.assert_called_once_with("00000000-0000-0000-0000-000000000004")


def test_authorize_transfer_uses_the_database_transition(client, mock_supabase, admin_token):
    """Status changes go through transfer_transition (one locked transaction, DB clock)."""
    transfer_id = "00000000-0000-0000-0000-000000000010"
    to_hosp = "00000000-0000-0000-0000-000000000013"
    row = {"transfer_id": transfer_id, "status": "pending_authorization", "accident_id": "00000000-0000-0000-0000-000000000011",
           "from_hospital_id": ADMIN_HOSPITAL, "to_hospital_id": to_hosp}
    mock_transfer_table = MagicMock()
    mock_transfer_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = MockDBResult(data=[row])
    mock_audit_table = MagicMock()

    def table_side_effect(table_name):
        if table_name == "User":
            mock_table = MagicMock()
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "00000000-0000-0000-0000-000000000002", "role": "admin", "hospital_id": ADMIN_HOSPITAL}
            ])
            mock_table.select.return_value.in_.return_value.execute.return_value = MockDBResult(data=[])
            return mock_table
        if table_name == "Transfer":
            return mock_transfer_table
        if table_name == "AuditLog":
            return mock_audit_table
        mock_table = MagicMock()
        mock_table.select.return_value.in_.return_value.execute.return_value = MockDBResult(data=[])
        return mock_table

    mock_supabase.table.side_effect = table_side_effect
    mock_supabase.rpc.return_value.execute.return_value = MockDBResult(data=[{**row, "status": "authorized"}])

    response = client.post(f"/api/admin/transfers/{transfer_id}/authorize",
                           headers={"Authorization": f"Bearer {admin_token}"}, json={"note": "Bed confirmed"})

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["status"] == "authorized"
    mock_supabase.rpc.assert_called_once()
    name, params = mock_supabase.rpc.call_args.args
    assert name == "transfer_transition"
    assert params["p_action"] == "authorize" and params["p_transfer_id"] == transfer_id
    assert params["p_actor"] == "00000000-0000-0000-0000-000000000002" and params["p_note"] == "Bed confirmed"
    mock_transfer_table.update.assert_not_called()   # no direct writes, no "now()" strings
    audit = mock_audit_table.insert.call_args.args[0]
    assert audit["action_type"] == "transfer_authorized"
    assert (audit["details"]["old_status"], audit["details"]["new_status"]) == ("pending_authorization", "authorized")


def test_the_old_free_form_status_endpoint_is_gone(client, mock_supabase, admin_token):
    mock_supabase.table.side_effect = _admin_tables()
    response = client.patch("/api/admin/transfers/00000000-0000-0000-0000-000000000010/status",
                            headers={"Authorization": f"Bearer {admin_token}"}, json={"status": "approved"})
    assert response.status_code in (404, 405)
    mock_supabase.rpc.assert_not_called()


@patch("app.api.admin.secrets.token_urlsafe", return_value="TempPass-generated")
def test_user_without_password_gets_a_temporary_one(mock_token, client, mock_supabase, admin_token):
    auth_user = MagicMock(id="00000000-0000-0000-0000-000000000099")
    mock_supabase.auth.admin.create_user.return_value = MagicMock(user=auth_user)
    insert_table = MagicMock()
    insert_table.insert.return_value.execute.return_value = MockDBResult(data=[{
        "user_id": auth_user.id, "name": "New Nurse", "email": "new@test.com", "role": "nurse",
        "hospital_id": ADMIN_HOSPITAL}])
    base = _admin_tables()

    def table(name):
        mock = base(name)
        if name == "User":
            mock.insert = insert_table.insert
        return mock

    mock_supabase.table.side_effect = table
    response = client.post("/api/admin/users", headers={"Authorization": f"Bearer {admin_token}"},
                           json={"email": "new@test.com", "name": "New Nurse", "role": "nurse"})
    assert response.status_code == status.HTTP_201_CREATED, response.text
    body = response.json()
    assert body["temporary_password"] == "TempPass-generated" and body["must_change_password"] is True
    sent = mock_supabase.auth.admin.create_user.call_args.args[0]
    assert sent["password"] == "TempPass-generated"
    assert sent["user_metadata"] == {"must_change_password": True}
    assert insert_table.insert.call_args.args[0]["hospital_id"] == ADMIN_HOSPITAL
    assert "password" not in insert_table.insert.call_args.args[0]


def test_admin_cannot_create_a_government_account(client, mock_supabase, admin_token):
    mock_supabase.table.side_effect = _admin_tables()
    response = client.post("/api/admin/users", headers={"Authorization": f"Bearer {admin_token}"},
                           json={"email": "g@test.com", "name": "G", "role": "government", "password": "SecurePassword123"})
    assert response.status_code == 422
    mock_supabase.auth.admin.create_user.assert_not_called()


def _admin_tables(transfer_rows=None, officer_rows=None):
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "00000000-0000-0000-0000-000000000002", "role": "admin", "hospital_id": ADMIN_HOSPITAL}
            ])
            mock_table.select.return_value.ilike.return_value.execute.return_value = MockDBResult(data=[])
        elif table_name == "Transfer":
            mock_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = MockDBResult(
                data=transfer_rows or [])
        elif table_name == "Government_Officer":
            mock_table.select.return_value.ilike.return_value.execute.return_value = MockDBResult(
                data=officer_rows or [])
        return mock_table
    return table_side_effect


def test_admin_cannot_create_user_for_another_hospital(client, mock_supabase, admin_token):
    mock_supabase.table.side_effect = _admin_tables()
    response = client.post("/api/admin/users", headers={"Authorization": f"Bearer {admin_token}"}, json={
        "email": "x@test.com", "password": "SecurePassword123", "name": "X", "role": "nurse",
        "hospital_id": "00000000-0000-0000-0000-0000000000b2",
    })
    assert response.status_code == status.HTTP_403_FORBIDDEN
    mock_supabase.auth.admin.create_user.assert_not_called()


def test_admin_cannot_take_over_a_government_login(client, mock_supabase, admin_token):
    mock_supabase.table.side_effect = _admin_tables(officer_rows=[{"officer_id": "gov-1"}])
    response = client.post("/api/admin/users", headers={"Authorization": f"Bearer {admin_token}"}, json={
        "email": "officer@health.gov.lk", "password": "SecurePassword123", "name": "X", "role": "nurse",
    })
    assert response.status_code == status.HTTP_409_CONFLICT
    mock_supabase.auth.admin.create_user.assert_not_called()
    mock_supabase.auth.admin.update_user_by_id.assert_not_called()


def test_admin_cannot_change_a_transfer_between_other_hospitals(client, mock_supabase, admin_token):
    mock_supabase.table.side_effect = _admin_tables(transfer_rows=[{
        "transfer_id": "00000000-0000-0000-0000-000000000010", "status": "pending_authorization",
        "from_hospital_id": "00000000-0000-0000-0000-0000000000b2",
        "to_hospital_id": "00000000-0000-0000-0000-0000000000c3",
    }])
    response = client.post(
        "/api/admin/transfers/00000000-0000-0000-0000-000000000010/authorize",
        headers={"Authorization": f"Bearer {admin_token}"}, json={},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND
    mock_supabase.rpc.assert_not_called()
