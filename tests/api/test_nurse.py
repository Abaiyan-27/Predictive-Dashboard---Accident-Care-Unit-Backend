from unittest.mock import MagicMock, patch
from fastapi import status
from app.services.schema_service import DatabaseMigrationRequired

class MockDBResult:
    def __init__(self, data):
        self.data = data

def setup_mock_db_for_nurse(mock_supabase, role: str):
    """Helper to mock DB response for get_current_user."""
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": f"00000000-0000-0000-0000-000000000001", "role": role, "hospital_id": "00000000-0000-0000-0000-000000000002"}
            ])
        return mock_table
    mock_supabase.table.side_effect = table_side_effect

@patch("app.api.nurse.schema_service.require_intake_schema")
@patch("app.api.nurse.patient_service.add_patient_predictions")
def test_create_patient_intake_success(mock_add_predictions, mock_require_schema, client, mock_supabase, mock_intake_payload, nurse_token):
    # Ensure current user is mocked properly
    setup_mock_db_for_nurse(mock_supabase, "nurse")
    
    # Setup mock tables for the intake flow
    patient_id = "00000000-0000-0000-0000-000000000010"
    accident_id = "00000000-0000-0000-0000-000000000020"
    injury_id = "00000000-0000-0000-0000-000000000030"
    
    mock_patient_table = MagicMock()
    mock_patient_table.insert.return_value.execute.return_value = MockDBResult(data=[{"patient_id": patient_id}])
    
    mock_accident_table = MagicMock()
    mock_accident_table.insert.return_value.execute.return_value = MockDBResult(data=[{"accident_id": accident_id}])
    
    mock_injury_table = MagicMock()
    mock_injury_table.insert.return_value.execute.return_value = MockDBResult(data=[{"injury_id": injury_id}])
    
    mock_hosp_patient_table = MagicMock()
    mock_hosp_patient_table.insert.return_value.execute.return_value = MockDBResult(data=[{"id": 1}])
    
    mock_audit_table = MagicMock()
    mock_audit_table.insert.return_value.execute.return_value = MockDBResult(data=[{"id": 1}])
    
    # We need to map `mock_supabase.table("Name")` to these specific mocks, plus "User" for auth.
    def table_side_effect(table_name):
        if table_name == "User":
            mock_table = MagicMock()
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": f"00000000-0000-0000-0000-000000000001", "role": "nurse", "hospital_id": "00000000-0000-0000-0000-000000000002"}
            ])
            return mock_table
        elif table_name == "Patient":
            return mock_patient_table
        elif table_name == "Accident_Record":
            return mock_accident_table
        elif table_name == "Injury_Description":
            return mock_injury_table
        elif table_name == "Hospital_Patient":
            return mock_hosp_patient_table
        elif table_name == "AuditLog":
            return mock_audit_table
        return MagicMock()
        
    mock_supabase.table.side_effect = table_side_effect
    
    # Mock the patient service prediction output
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
    data = response.json()
    assert data["patient"]["patient_id"] == patient_id
    assert data["accident_record"]["accident_id"] == accident_id
    assert data["severity_prediction"] == "critical"
    
    mock_require_schema.assert_called_once_with(mock_supabase)
    mock_patient_table.insert.assert_called_once()
    mock_accident_table.insert.assert_called_once()
    mock_injury_table.insert.assert_called_once()
    mock_hosp_patient_table.insert.assert_called_once()
    mock_audit_table.insert.assert_called_once()
    mock_add_predictions.assert_called_once()


@patch("app.api.nurse.schema_service.require_intake_schema")
def test_create_patient_intake_migration_required(mock_require_schema, client, mock_supabase, mock_intake_payload, nurse_token):
    setup_mock_db_for_nurse(mock_supabase, "nurse")
    
    mock_require_schema.side_effect = DatabaseMigrationRequired("Accident_Record", "Column missing")

    response = client.post(
        "/api/patients/intake",
        headers={"Authorization": f"Bearer {nurse_token}"},
        json=mock_intake_payload
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "Database setup is incomplete" in response.json()["detail"]


@patch("app.api.nurse.schema_service.require_intake_schema")
def test_create_patient_intake_rollback_on_accident_failure(mock_require_schema, client, mock_supabase, mock_intake_payload, nurse_token):
    patient_id = "00000000-0000-0000-0000-000000000010"
    
    mock_patient_table = MagicMock()
    mock_patient_table.insert.return_value.execute.return_value = MockDBResult(data=[{"patient_id": patient_id}])
    
    mock_accident_table = MagicMock()
    # Simulate DB error on accident record insert
    mock_accident_table.insert.return_value.execute.side_effect = Exception("DB Insert Failed")
    
    def table_side_effect(table_name):
        if table_name == "User":
            mock_table = MagicMock()
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": f"00000000-0000-0000-0000-000000000001", "role": "nurse", "hospital_id": "00000000-0000-0000-0000-000000000002"}
            ])
            return mock_table
        elif table_name == "Patient":
            return mock_patient_table
        elif table_name == "Accident_Record":
            return mock_accident_table
        return MagicMock()
        
    mock_supabase.table.side_effect = table_side_effect

    import pytest
    with pytest.raises(Exception, match="DB Insert Failed"):
        client.post(
            "/api/patients/intake",
            headers={"Authorization": f"Bearer {nurse_token}"},
            json=mock_intake_payload
        )
    
    # Verify rollback was called on Patient table
    mock_patient_table.delete.assert_called_once()
    mock_patient_table.delete.return_value.eq.assert_called_once_with("patient_id", patient_id)
    mock_patient_table.delete.return_value.eq.return_value.execute.assert_called_once()


@patch("app.api.nurse.patient_service.add_severity_predictions")
def test_get_my_patients(mock_add_severity, client, mock_supabase, nurse_token):
    mock_accident_table = MagicMock()
    mock_accident_table.select.return_value.eq.return_value.order.return_value.execute.return_value = MockDBResult(data=[
        {"accident_id": "00000000-0000-0000-0000-000000000020", "created_by": "00000000-0000-0000-0000-000000000001"}
    ])
    
    def table_side_effect(table_name):
        if table_name == "User":
            mock_table = MagicMock()
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": f"00000000-0000-0000-0000-000000000001", "role": "nurse"}
            ])
            return mock_table
        elif table_name == "Accident_Record":
            return mock_accident_table
        return MagicMock()
        
    mock_supabase.table.side_effect = table_side_effect
    
    mock_add_severity.return_value = [{"accident_id": "00000000-0000-0000-0000-000000000020"}]

    response = client.get(
        "/api/patients/my",
        headers={"Authorization": f"Bearer {nurse_token}"},
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) == 1
    
    mock_accident_table.select.assert_called_once_with("*, Patient(*), Injury_Description(*), Treatment(*), Transfer(*)")
    mock_accident_table.select.return_value.eq.assert_called_once_with("created_by", "00000000-0000-0000-0000-000000000001")
