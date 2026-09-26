import pytest
from unittest.mock import MagicMock
from fastapi import status

class MockDBResult:
    def __init__(self, data):
        self.data = data

def test_get_hospitals(client, mock_supabase, admin_token):
    mock_supabase.table.return_value.select.return_value.execute.return_value = MockDBResult(data=[
        {"hospital_id": "00000000-0000-0000-0000-000000000001", "name": "General Hospital"}
    ])
    
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "00000000-0000-0000-0000-000000000002", "role": "admin"}
            ])
        elif table_name == "Hospital":
            mock_table.select.return_value.execute.return_value = MockDBResult(data=[
                {"hospital_id": "00000000-0000-0000-0000-000000000001", "name": "General Hospital"}
            ])
        return mock_table
        
    mock_supabase.table.side_effect = table_side_effect

    response = client.get(
        "/api/admin/hospitals",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()) == 1


def test_create_hospital(client, mock_supabase, admin_token):
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
                {"user_id": "00000000-0000-0000-0000-000000000002", "role": "admin"}
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
    
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["hospital_id"] == "00000000-0000-0000-0000-000000000003"
    mock_hosp_table.insert.assert_called_once()
    mock_audit_table.insert.assert_called_once()


def test_create_user_success(client, mock_supabase, admin_token):
    mock_auth_result = MagicMock()
    mock_auth_result.user.id = "00000000-0000-0000-0000-000000000004"
    mock_supabase.auth.admin.create_user.return_value = mock_auth_result
    
    mock_user_table = MagicMock()
    mock_user_table.select.return_value.eq.return_value.execute.side_effect = [
        MockDBResult(data=[{"user_id": "00000000-0000-0000-0000-000000000002", "role": "admin"}]), # deps.py check
        MockDBResult(data=[]) # user exists check
    ]
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


def test_create_user_rollback_on_profile_fail(client, mock_supabase, admin_token):
    mock_auth_result = MagicMock()
    mock_auth_result.user.id = "00000000-0000-0000-0000-000000000004"
    mock_supabase.auth.admin.create_user.return_value = mock_auth_result
    
    mock_user_table = MagicMock()
    mock_user_table.select.return_value.eq.return_value.execute.side_effect = [
        MockDBResult(data=[{"user_id": "00000000-0000-0000-0000-000000000002", "role": "admin"}]), 
        MockDBResult(data=[])
    ]
    # Simulate DB failure on profile insert
    mock_user_table.insert.return_value.execute.side_effect = Exception("DB Fail")
    
    def table_side_effect(table_name):
        if table_name == "User":
            return mock_user_table
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


def test_update_transfer_status_approved(client, mock_supabase, admin_token):
    # Tests the multi-step approval workflow
    transfer_id = "00000000-0000-0000-0000-000000000010"
    accident_id = "00000000-0000-0000-0000-000000000011"
    from_hosp = "00000000-0000-0000-0000-000000000012"
    to_hosp = "00000000-0000-0000-0000-000000000013"
    
    mock_transfer_table = MagicMock()
    mock_transfer_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
        {"transfer_id": transfer_id, "status": "pending_authorization"}
    ])
    mock_transfer_table.update.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
        {
            "transfer_id": transfer_id, 
            "status": "approved", 
            "accident_id": accident_id,
            "from_hospital_id": from_hosp,
            "to_hospital_id": to_hosp
        }
    ])
    
    mock_accident_table = MagicMock()
    mock_hosp_patient_table = MagicMock()
    mock_audit_table = MagicMock()
    
    def table_side_effect(table_name):
        if table_name == "User":
            mock_table = MagicMock()
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "00000000-0000-0000-0000-000000000002", "role": "admin"}
            ])
            return mock_table
        elif table_name == "Transfer":
            return mock_transfer_table
        elif table_name == "Accident_Record":
            return mock_accident_table
        elif table_name == "Hospital_Patient":
            return mock_hosp_patient_table
        elif table_name == "AuditLog":
            return mock_audit_table
        return MagicMock()
        
    mock_supabase.table.side_effect = table_side_effect
    
    response = client.patch(
        f"/api/admin/transfers/{transfer_id}/status",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"status": "approved"}
    )
    
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "approved"
    
    # Verify Accident_Record updated
    mock_accident_table.update.assert_called_once_with({"hospital_id": to_hosp})
    
    # Verify Hospital_Patient closed out old record
    mock_hosp_patient_table.update.assert_called_once_with({"exited_at": "now()"})
    
    # Verify Hospital_Patient inserted new record
    mock_hosp_patient_table.insert.assert_called_once_with({
        "accident_id": accident_id,
        "hospital_id": to_hosp
    })
    
    # Verify Audit log
    mock_audit_table.insert.assert_called_once()
