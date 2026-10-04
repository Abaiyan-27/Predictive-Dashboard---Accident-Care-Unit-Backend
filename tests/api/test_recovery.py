import pytest
from fastapi import status
from unittest.mock import MagicMock, patch

class MockDBResult:
    def __init__(self, data, count=None):
        self.data = data
        self.count = len(data) if count is None and isinstance(data, list) else count

def setup_mock_db_for_nurse(mock_supabase, role: str):
    """Helper to mock DB response for get_current_user."""
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "nurse-123", "role": role, "hospital_id": "00000000-0000-0000-0000-000000000002"}
            ])
        return mock_table
    mock_supabase.table.side_effect = table_side_effect


@patch("app.api.nurse.schema_service.require_intake_schema")
def test_rec_001_client_interruption_rollback(mock_require_schema, client, mock_supabase, mock_intake_payload, nurse_token):
    """
    REC-001: Incomplete transaction.
    Failure condition: Force a failure during the second step of a multi-step database operation (Accident_Record insert).
    Expected Recovery: Verify that the partially committed Patient record is rolled back explicitly by the app.
    """
    patient_id = "00000000-0000-0000-0000-000000000010"
    
    mock_patient_table = MagicMock()
    mock_patient_table.insert.return_value.execute.return_value = MockDBResult(data=[{"patient_id": patient_id}])
    
    mock_accident_table = MagicMock()
    # Simulate DB error on accident record insert
    mock_accident_table.insert.return_value.execute.side_effect = Exception("DB Connection Dropped Mid-Transaction")
    
    def table_side_effect(table_name):
        if table_name == "User":
            mock_table = MagicMock()
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "nurse-123", "role": "nurse", "hospital_id": "00000000-0000-0000-0000-000000000002"}
            ])
            return mock_table
        elif table_name == "Patient":
            return mock_patient_table
        elif table_name == "Accident_Record":
            return mock_accident_table
        return MagicMock()
        
    mock_supabase.table.side_effect = table_side_effect

    # The exception should be propagated back as a 500
    with pytest.raises(Exception, match="DB Connection Dropped Mid-Transaction"):
        client.post(
            "/api/patients/intake",
            headers={"Authorization": f"Bearer {nurse_token}"},
            json=mock_intake_payload
        )
    
    # VERIFY EXPECTED RECOVERY BEHAVIOR:
    # Verify rollback was explicitly called on the Patient table because Supabase REST API doesn't support implicit transactions
    mock_patient_table.delete.assert_called_once()
    mock_patient_table.delete.return_value.eq.assert_called_once_with("patient_id", patient_id)
    mock_patient_table.delete.return_value.eq.return_value.execute.assert_called_once()


@patch("app.api.nurse.patient_service.add_severity_predictions")
def test_rec_002_and_003_network_interruption_and_rbac_recovery(mock_add_severity, client, mock_supabase, nurse_token):
    """
    REC-002: Network Interruption / Database Restart
    Failure condition: Database becomes unreachable temporarily.
    Expected Recovery: Application returns a 500 error gracefully, and when DB is restored, queries succeed again.
    
    REC-003: Recovery and Access Control
    Failure condition: After the database is restored in REC-002.
    Expected Recovery: Authentication and RBAC still strictly enforce permissions.
    """
    
    # -------------------------------------------------------------
    # PHASE 1: Simulate Network Interruption
    # -------------------------------------------------------------
    def broken_table_side_effect(table_name):
        if table_name == "User":
            mock_table = MagicMock()
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "nurse-123", "role": "nurse", "hospital_id": "00000000-0000-0000-0000-000000000002"}
            ])
            return mock_table
        
        # All other queries fail because the network dropped
        mock_table = MagicMock()
        mock_table.select.return_value.eq.return_value.order.return_value.range.return_value.execute.side_effect = Exception("Network Reset By Peer")
        return mock_table
        
    mock_supabase.table.side_effect = broken_table_side_effect
    
    with pytest.raises(Exception, match="Network Reset By Peer"):
        client.get(
            "/api/patients/my",
            headers={"Authorization": f"Bearer {nurse_token}"},
        )
    
    # -------------------------------------------------------------
    # PHASE 2: Simulate Database Restoration
    # -------------------------------------------------------------
    mock_accident_table = MagicMock()
    mock_accident_table.select.return_value.eq.return_value.order.return_value.range.return_value.execute.return_value = MockDBResult(data=[
        {"accident_id": "a-123", "created_by": "nurse-123"}
    ])
    mock_add_severity.return_value = [{"accident_id": "a-123"}]

    def fixed_table_side_effect(table_name):
        if table_name == "User":
            mock_table = MagicMock()
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "nurse-123", "role": "nurse", "hospital_id": "00000000-0000-0000-0000-000000000002"}
            ])
            return mock_table
        elif table_name == "Accident_Record":
            return mock_accident_table
        return MagicMock()

    mock_supabase.table.side_effect = fixed_table_side_effect
    
    # Test that the app recovered and serves data again
    response = client.get(
        "/api/patients/my",
        headers={"Authorization": f"Bearer {nurse_token}"},
    )
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()) == 1
    
    # -------------------------------------------------------------
    # PHASE 3: REC-003 Verify RBAC Post-Recovery
    # -------------------------------------------------------------
    # Nurse attempts to hit an Admin route after DB recovery
    response = client.get(
        "/api/admin/users",
        headers={"Authorization": f"Bearer {nurse_token}"}
    )
    # Role checker still works perfectly!
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert "Access denied" in response.json()["detail"]
