from unittest.mock import MagicMock, patch
from fastapi import status
from app.services.schema_service import DatabaseMigrationRequired

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
                {"user_id": f"00000000-0000-0000-0000-000000000001", "role": role, "hospital_id": "00000000-0000-0000-0000-000000000002"}
            ])
        return mock_table
    mock_supabase.table.side_effect = table_side_effect

@patch("app.api.nurse.schema_service.require_intake_schema")
@patch("app.api.nurse.patient_service.predict_and_store")
@patch("app.services.patient_service.run_models")
def test_create_patient_intake_success(mock_run_models, mock_predict_and_store, mock_require_schema, client, mock_supabase, mock_intake_payload, nurse_token):
    # Ensure current user is mocked properly
    setup_mock_db_for_nurse(mock_supabase, "nurse")
    
    # Setup mock tables for the intake flow
    patient_id = "00000000-0000-0000-0000-000000000010"
    accident_id = "00000000-0000-0000-0000-000000000020"
    injury_id = "00000000-0000-0000-0000-000000000030"
    
    mock_patient_table = MagicMock()
    mock_patient_table.insert.side_effect = lambda row: MagicMock(
        execute=MagicMock(return_value=MockDBResult(data=[{**row, "patient_id": patient_id}]))
    )
    
    mock_accident_table = MagicMock()
    # Inserts echo the written row back, as PostgREST does.
    mock_accident_table.insert.side_effect = lambda row: MagicMock(
        execute=MagicMock(return_value=MockDBResult(data=[{**row, "accident_id": accident_id}]))
    )
    
    mock_injury_table = MagicMock()
    mock_injury_table.insert.side_effect = lambda row: MagicMock(
        execute=MagicMock(return_value=MockDBResult(data=[{**row, "injury_id": injury_id}]))
    )
    
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
    
    response = client.post(
        "/api/patients/intake",
        headers={"Authorization": f"Bearer {nurse_token}"},
        json=mock_intake_payload
    )

    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["patient"]["patient_id"] == patient_id
    assert data["accident_record"]["accident_id"] == accident_id
    # Saving intake runs no model: Severity waits for the nurse to run it, and
    # the stay/transfer/discharge models are never run from the nurse workflow.
    assert "severity_prediction" not in data
    assert "stay_duration_prediction" not in data and "transfer_prediction" not in data
    assert data["severity"]["state"] == "NOT_RUN"
    assert data["severity"]["prediction"] is None and data["severity"]["decision"] is None
    assert data["severity"]["can_run"] is True
    
    mock_require_schema.assert_called_once_with(mock_supabase)
    mock_patient_table.insert.assert_called_once()
    mock_accident_table.insert.assert_called_once()
    mock_injury_table.insert.assert_called_once()
    assert mock_injury_table.insert.call_args.args[0]["injury_order"] == 1
    mock_hosp_patient_table.insert.assert_called_once()
    mock_audit_table.insert.assert_called_once()
    mock_predict_and_store.assert_not_called()
    mock_run_models.assert_not_called()


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
    mock_accident_table.select.return_value.eq.return_value.order.return_value.range.return_value.execute.return_value = MockDBResult(data=[
        {"accident_id": "00000000-0000-0000-0000-000000000020", "created_by": "00000000-0000-0000-0000-000000000099",
         "Patient": {"full_name": "A", "age": "30", "gender": "Male"}, "Injury_Description": [], "Treatment": []}
    ])
    mock_prediction_table = MagicMock()
    mock_prediction_table.select.return_value.in_.return_value.eq.return_value.order.return_value.execute.return_value = MockDBResult(data=[])
    
    def table_side_effect(table_name):
        if table_name == "User":
            mock_table = MagicMock()
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": f"00000000-0000-0000-0000-000000000001", "role": "nurse",
                 "hospital_id": "00000000-0000-0000-0000-000000000002"}
            ])
            return mock_table
        elif table_name == "Accident_Record":
            return mock_accident_table
        elif table_name == "Model_Prediction":
            return mock_prediction_table
        return MagicMock()
        
    mock_supabase.table.side_effect = table_side_effect

    response = client.get(
        "/api/patients/my",
        headers={"Authorization": f"Bearer {nurse_token}"},
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) == 1
    
    mock_accident_table.select.assert_called_once_with(
        "*, Patient(*), Injury_Description(*), Treatment(*), Transfer(*)", count="exact")
    # Paged: the first 50 rows, and the full size in the header.
    mock_accident_table.select.return_value.eq.return_value.order.return_value.range.assert_called_once_with(0, 49)
    assert response.headers["X-Total-Count"] == "1"
    # Visibility is the whole hospital, not the records this nurse created.
    mock_accident_table.select.return_value.eq.assert_called_once_with("hospital_id", "00000000-0000-0000-0000-000000000002")
    # Nurses get the stored Severity workflow only: no model is run on read and
    # no stay/transfer output is attached.
    mock_add_severity.assert_not_called()
    assert "severity_prediction" not in data[0]
    assert "stay_duration_prediction" not in data[0] and "transfer_prediction" not in data[0]
    assert data[0]["severity_workflow"]["state"] in {"NOT_RUN", "WAITING_FOR_INPUTS"}
    assert data[0]["severity_workflow"]["basic_complete"] is True


@patch("app.api.nurse.schema_service.require_intake_schema")
def test_staff_without_hospital_cannot_create_patients(mock_require_schema, client, mock_supabase, nurse_token):
    patient_table = MagicMock()

    def table_side_effect(table_name):
        if table_name == "User":
            mock_table = MagicMock()
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "nurse-1", "role": "nurse", "hospital_id": None}
            ])
            return mock_table
        return patient_table

    mock_supabase.table.side_effect = table_side_effect
    response = client.post(
        "/api/patients/intake",
        headers={"Authorization": f"Bearer {nurse_token}"},
        json={"full_name": "No hospital"},
    )
    assert response.status_code == 403
    assert "not assigned to a hospital" in response.json()["detail"]
    patient_table.insert.assert_not_called()


@patch("app.api.nurse.schema_service.require_intake_schema")
def test_intake_for_another_hospital_is_rejected(mock_require_schema, client, mock_supabase, nurse_token):
    setup_mock_db_for_nurse(mock_supabase, "nurse")
    response = client.post(
        "/api/patients/intake",
        headers={"Authorization": f"Bearer {nurse_token}"},
        json={"full_name": "X", "hospital_id": "00000000-0000-0000-0000-0000000000b2"},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "hospital_id must be your own hospital."


def test_intake_rejects_more_than_two_injuries(client, mock_supabase, nurse_token):
    setup_mock_db_for_nurse(mock_supabase, "nurse")
    response = client.post(
        "/api/patients/intake",
        headers={"Authorization": f"Bearer {nurse_token}"},
        json={"full_name": "X", "injuries": [{"site_of_injury": "a"}, {"site_of_injury": "b"}, {"site_of_injury": "c"}]},
    )
    assert response.status_code == 422
