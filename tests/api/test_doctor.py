from unittest.mock import MagicMock, patch
from fastapi import status

class MockDBResult:
    def __init__(self, data):
        self.data = data

def test_get_hospital_patients_success(client, mock_supabase, doctor_token):
    # Doctor is assigned to hosp-123
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "doctor-id-123", "role": "doctor", "hospital_id": "hosp-123"}
            ])
        elif table_name == "Accident_Record":
            mock_table.select.return_value.eq.return_value.order.return_value.execute.return_value = MockDBResult(data=[
                {"accident_id": "acc-1"}
            ])
        return mock_table
        
    mock_supabase.table.side_effect = table_side_effect
    
    with patch("app.api.doctor.patient_service.add_severity_predictions") as mock_add_predictions:
        mock_add_predictions.return_value = [{"accident_id": "acc-1", "severity_prediction": "Moderate"}]
        
        response = client.get(
            "/api/patients/hospital/hosp-123",
            headers={"Authorization": f"Bearer {doctor_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["severity_prediction"] == "Moderate"


def test_get_hospital_patients_bola_rejection(client, mock_supabase, doctor_token):
    # Doctor is assigned to hosp-123, but tries to fetch hosp-999
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "doctor-id-123", "role": "doctor", "hospital_id": "hosp-123"}
            ])
        return mock_table
        
    mock_supabase.table.side_effect = table_side_effect
    
    response = client.get(
        "/api/patients/hospital/hosp-999",
        headers={"Authorization": f"Bearer {doctor_token}"}
    )
    
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert "Doctors can only view patients assigned to their hospital" in response.json()["detail"]


def test_get_patient_not_found(client, mock_supabase, doctor_token):
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "doctor-id-123", "role": "doctor", "hospital_id": "hosp-123"}
            ])
        elif table_name == "Patient":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[])
        return mock_table
        
    mock_supabase.table.side_effect = table_side_effect
    
    response = client.get(
        "/api/patients/00000000-0000-0000-0000-000000000001",
        headers={"Authorization": f"Bearer {doctor_token}"}
    )
    
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_create_treatment(client, mock_supabase, doctor_token):
    mock_treatment_table = MagicMock()
    mock_treatment_table.insert.return_value.execute.return_value = MockDBResult(data=[
        {"treatment_id": "treat-1"}
    ])
    
    mock_audit_table = MagicMock()
    mock_audit_table.insert.return_value.execute.return_value = MockDBResult(data=[{"id": 1}])
    
    def table_side_effect(table_name):
        if table_name == "User":
            mock_table = MagicMock()
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "doctor-id-123", "role": "doctor", "hospital_id": "hosp-123"}
            ])
            return mock_table
        elif table_name == "Treatment":
            return mock_treatment_table
        elif table_name == "AuditLog":
            return mock_audit_table
        return MagicMock()
        
    mock_supabase.table.side_effect = table_side_effect
    
    payload = {
        "accident_id": "00000000-0000-0000-0000-000000000010",
        "hospital_id": "00000000-0000-0000-0000-000000000020",
        "ward_number": "ICU-1",
        "number_of_days_stay": 5,
        "investigation_done": "CT Scan",
        "management_name_1": "Surgery"
    }

    response = client.post(
        "/api/treatments",
        headers={"Authorization": f"Bearer {doctor_token}"},
        json=payload
    )
    
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["treatment_id"] == "treat-1"
    
    mock_treatment_table.insert.assert_called_once()
    mock_audit_table.insert.assert_called_once()


@patch("app.api.doctor.ml_service.predict_severity")
@patch("app.api.doctor.ml_service.predict_stay_duration")
@patch("app.api.doctor.ml_service.predict_transfer_probability")
@patch("app.api.doctor.ml_service.predict_discharge_outcome")
def test_get_patient_prognosis(
    mock_predict_discharge, 
    mock_predict_transfer, 
    mock_predict_stay, 
    mock_predict_severity, 
    client, 
    mock_supabase, 
    doctor_token
):
    # Setup mocks to return a dummy to_dict() object
    mock_prediction_result = MagicMock()
    mock_prediction_result.to_dict.return_value = {"prediction": "High"}
    
    mock_predict_severity.return_value = mock_prediction_result
    mock_predict_stay.return_value = mock_prediction_result
    mock_predict_transfer.return_value = mock_prediction_result
    mock_predict_discharge.return_value = mock_prediction_result
    
    mock_accident_table = MagicMock()
    mock_accident_table.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MockDBResult(data=[
        {"accident_id": "acc-1"}
    ])
    
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "doctor-id-123", "role": "doctor", "hospital_id": "hosp-123"}
            ])
        elif table_name == "Accident_Record":
            return mock_accident_table
        elif table_name == "Patient":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"patient_id": "00000000-0000-0000-0000-000000000001"}
            ])
        elif table_name == "Transfer":
            mock_table.select.return_value.eq.return_value.order.return_value.execute.return_value = MockDBResult(data=[])
        return mock_table
        
    mock_supabase.table.side_effect = table_side_effect
    
    response = client.get(
        "/api/patients/00000000-0000-0000-0000-000000000001/prognosis",
        headers={"Authorization": f"Bearer {doctor_token}"}
    )
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["severity"]["prediction"] == "High"
    assert data["stay_duration"]["prediction"] == "High"
    
    # Assert all 4 models were called exactly once
    mock_predict_severity.assert_called_once()
    mock_predict_stay.assert_called_once()
    mock_predict_transfer.assert_called_once()
    mock_predict_discharge.assert_called_once()


def test_get_patient_prognosis_no_accident_record(client, mock_supabase, doctor_token):
    mock_accident_table = MagicMock()
    # Return empty list simulating no accident record found
    mock_accident_table.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MockDBResult(data=[])
    
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "doctor-id-123", "role": "doctor", "hospital_id": "hosp-123"}
            ])
        elif table_name == "Accident_Record":
            return mock_accident_table
        return mock_table
        
    mock_supabase.table.side_effect = table_side_effect
    
    response = client.get(
        "/api/patients/00000000-0000-0000-0000-000000000001/prognosis",
        headers={"Authorization": f"Bearer {doctor_token}"}
    )
    
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "No accident record found" in response.json()["detail"]
