from unittest.mock import MagicMock, patch
from fastapi import status

class MockDBResult:
    def __init__(self, data, count=None):
        self.data = data
        self.count = len(data) if count is None and isinstance(data, list) else count


HOSPITAL_A = "00000000-0000-0000-0000-0000000000a1"
HOSPITAL_B = "00000000-0000-0000-0000-0000000000b2"

def test_get_hospital_patients_success(client, mock_supabase, doctor_token):
    # Doctor is assigned to hosp-123
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "doctor-id-123", "role": "doctor", "hospital_id": HOSPITAL_A}
            ])
        elif table_name == "Accident_Record":
            mock_table.select.return_value.eq.return_value.order.return_value.range.return_value.execute.return_value = MockDBResult(data=[
                {"accident_id": "acc-1", "hospital_id": HOSPITAL_A}
            ])
        return mock_table
        
    mock_supabase.table.side_effect = table_side_effect
    
    with patch("app.api.doctor.patient_service.add_severity_predictions") as mock_add_predictions, \
            patch("app.api.doctor.severity_workflow.statuses_for") as severity_statuses, \
            patch("app.api.doctor.doctor_workflow.summaries_for") as doctor_summaries:
        severity_statuses.return_value = {"acc-1": {"state": "ACCEPTED", "final_severity": "BORDERLINE"}}
        doctor_summaries.return_value = {"acc-1": {"treatment_recorded": False, "models": {}}}

        response = client.get(
            f"/api/patients/hospital/{HOSPITAL_A}",
            headers={"Authorization": f"Bearer {doctor_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        # Doctors get the stored state only; no model is run on read.
        mock_add_predictions.assert_not_called()
        assert data[0]["severity_workflow"]["final_severity"] == "BORDERLINE"
        assert data[0]["doctor_workflow"] == {"treatment_recorded": False, "models": {}}
        assert "severity_prediction" not in data[0]


def test_get_hospital_patients_bola_rejection(client, mock_supabase, doctor_token):
    # Doctor is assigned to hosp-123, but tries to fetch hosp-999
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "doctor-id-123", "role": "doctor", "hospital_id": HOSPITAL_A}
            ])
        return mock_table
        
    mock_supabase.table.side_effect = table_side_effect
    
    response = client.get(
        f"/api/patients/hospital/{HOSPITAL_B}",
        headers={"Authorization": f"Bearer {doctor_token}"}
    )
    
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert response.json()["detail"] == "hospital_id must be your own hospital."


def test_get_patient_not_found(client, mock_supabase, doctor_token):
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "doctor-id-123", "role": "doctor", "hospital_id": HOSPITAL_A}
            ])
        elif table_name == "Accident_Record":
            mock_table.select.return_value.eq.return_value.eq.return_value.order.return_value.execute.return_value = MockDBResult(data=[])
        return mock_table
        
    mock_supabase.table.side_effect = table_side_effect
    
    response = client.get(
        "/api/patients/00000000-0000-0000-0000-000000000001",
        headers={"Authorization": f"Bearer {doctor_token}"}
    )
    
    assert response.status_code == status.HTTP_404_NOT_FOUND


def _treatment_tables(mock_supabase, mock_treatment_table, mock_audit_table, role="nurse"):
    """Treatment is recorded by a nurse (every model input comes from a nurse)."""
    def table_side_effect(table_name):
        if table_name == "User":
            mock_table = MagicMock()
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": f"{role}-id-123", "role": role, "hospital_id": HOSPITAL_A}
            ])
            return mock_table
        elif table_name == "Treatment":
            return mock_treatment_table
        elif table_name == "AuditLog":
            return mock_audit_table
        elif table_name == "Accident_Record":
            accident_table = MagicMock()
            accident_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = MockDBResult(
                data=[{"accident_id": "00000000-0000-0000-0000-000000000010", "hospital_id": HOSPITAL_A,
                       "patient_id": "00000000-0000-0000-0000-000000000001", "Treatment": []}])
            return accident_table
        return MagicMock()

    mock_supabase.table.side_effect = table_side_effect


TREATMENT_PAYLOAD = {
    "accident_id": "00000000-0000-0000-0000-000000000010",
    "ward_number": "ICU-1",
    "number_of_days_stay": 5,
    "investigation_done": "CT Scan",
    "management_name_1": "Surgery",
}
UNDECIDED = {key: {"decision": None} for key in ("stay_duration", "transfer", "discharge")}


@patch("app.api.doctor.doctor_workflow.run_models")
@patch("app.api.doctor.doctor_workflow.load_models_status", return_value=UNDECIDED)
def test_create_treatment(mock_status, mock_run_models, client, mock_supabase, nurse_token):
    mock_treatment_table = MagicMock()
    mock_treatment_table.insert.return_value.execute.return_value = MockDBResult(data=[
        {"treatment_id": "treat-1"}
    ])
    mock_audit_table = MagicMock()
    mock_audit_table.insert.return_value.execute.return_value = MockDBResult(data=[{"id": 1}])
    _treatment_tables(mock_supabase, mock_treatment_table, mock_audit_table)

    # No hospital_id is sent: the nurse's own hospital is used.
    response = client.post(
        "/api/treatments",
        headers={"Authorization": f"Bearer {nurse_token}"},
        json=TREATMENT_PAYLOAD,
    )

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["treatment_id"] == "treat-1"
    assert "doctor_workflow" in response.json()

    mock_treatment_table.insert.assert_called_once()
    assert mock_treatment_table.insert.call_args.args[0]["hospital_id"] == HOSPITAL_A
    assert mock_treatment_table.insert.call_args.args[0]["logged_by"] == "nurse-id-123"
    assert mock_treatment_table.insert.call_args.args[0]["decided_by"] is None
    mock_audit_table.insert.assert_called_once()
    audit = mock_audit_table.insert.call_args.args[0]
    assert audit["action_type"] == "treatment_logged" and audit["details"]["treatment_id"] == "treat-1"
    assert mock_run_models.call_args.kwargs["trigger_event"] == "treatment_recorded"


@patch("app.api.doctor.doctor_workflow.run_models")
@patch("app.api.doctor.doctor_workflow.load_models_status",
       return_value={**UNDECIDED, "transfer": {"decision": {"final_decision": "Transfer"}}})
def test_treatment_is_locked_after_a_doctor_decision(mock_status, mock_run_models, client, mock_supabase, nurse_token):
    mock_treatment_table = MagicMock()
    _treatment_tables(mock_supabase, mock_treatment_table, MagicMock())
    response = client.post(
        "/api/treatments",
        headers={"Authorization": f"Bearer {nurse_token}"},
        json=TREATMENT_PAYLOAD,
    )
    assert response.status_code == status.HTTP_409_CONFLICT
    mock_treatment_table.insert.assert_not_called()
    mock_run_models.assert_not_called()


def test_treatment_for_another_hospital_id_is_refused(client, mock_supabase, nurse_token):
    mock_treatment_table = MagicMock()
    _treatment_tables(mock_supabase, mock_treatment_table, MagicMock())
    response = client.post(
        "/api/treatments",
        headers={"Authorization": f"Bearer {nurse_token}"},
        json={**TREATMENT_PAYLOAD, "hospital_id": HOSPITAL_B},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN
    mock_treatment_table.insert.assert_not_called()


def test_doctors_cannot_record_treatment(client, mock_supabase, doctor_token):
    """Doctors review predictions and decide; they do not enter model inputs."""
    mock_treatment_table = MagicMock()
    _treatment_tables(mock_supabase, mock_treatment_table, MagicMock(), role="doctor")
    response = client.post("/api/treatments", headers={"Authorization": f"Bearer {doctor_token}"}, json=TREATMENT_PAYLOAD)
    assert response.status_code == status.HTTP_403_FORBIDDEN
    mock_treatment_table.insert.assert_not_called()


@patch("app.api.doctor.severity_workflow.load_status")
@patch("app.api.doctor.prediction_store.store_predictions")
@patch("app.api.doctor.ml_service.bundle_readiness")
@patch("app.api.doctor.patient_service.run_models")
def test_get_patient_prognosis(mock_run_models, mock_readiness, mock_store, mock_severity, client, mock_supabase, doctor_token):
    mock_run_models.return_value = {
        "stay_duration": {"prediction": "2-3 days", "status": "OK"},
        "transfer": {"prediction": None, "status": "REFUSED"},
        "discharge": {"prediction": "Complete recovery", "status": "OK"},
    }
    mock_readiness.side_effect = lambda key, record: {"ready": key != "transfer"}
    mock_severity.return_value = {"prediction": {"prediction": "HIGH", "prediction_id": "p-sev"}}

    def stored(supabase, **kwargs):
        for key, output in kwargs["predictions"].items():
            output["prediction_id"] = f"p-{key}"
    mock_store.side_effect = stored

    mock_accident_table = MagicMock()
    mock_accident_table.select.return_value.eq.return_value.eq.return_value.order.return_value.execute.return_value = MockDBResult(data=[
        {"accident_id": "acc-1", "hospital_id": HOSPITAL_A, "patient_id": "00000000-0000-0000-0000-000000000001"}
    ])

    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "doctor-id-123", "role": "doctor", "hospital_id": HOSPITAL_A}
            ])
        elif table_name == "Accident_Record":
            return mock_accident_table
        elif table_name == "Patient":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"patient_id": "00000000-0000-0000-0000-000000000001"}
            ])
        return mock_table

    mock_supabase.table.side_effect = table_side_effect

    response = client.get(
        "/api/patients/00000000-0000-0000-0000-000000000001/prognosis",
        headers={"Authorization": f"Bearer {doctor_token}"}
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # Severity is read from storage, never run by the doctor.
    assert data["severity"]["prediction_id"] == "p-sev"
    assert mock_run_models.call_args.args[1] == ("stay_duration", "transfer", "discharge")
    assert data["stay_duration"]["prediction"] == "2-3 days"
    # Only models whose inputs are complete are stored.
    kwargs = mock_store.call_args.kwargs
    assert set(kwargs["predictions"]) == {"stay_duration", "discharge"}
    assert kwargs["trigger_event"] == "prognosis_viewed"
    assert kwargs["requested_by"] == "doctor-id-123"


@patch("app.api.doctor.severity_workflow.load_status")
@patch("app.api.doctor.prediction_store.store_predictions")
@patch("app.api.doctor.ml_service.bundle_readiness")
@patch("app.api.doctor.patient_service.run_models")
def test_prognosis_refuses_to_return_a_prediction_that_was_not_saved(mock_run_models, mock_readiness, mock_store, mock_severity, client, mock_supabase, doctor_token):
    mock_run_models.return_value = {
        "stay_duration": {"prediction": "2-3 days", "status": "OK"},
        "transfer": {"prediction": None, "status": "REFUSED"},
        "discharge": {"prediction": "Complete recovery", "status": "OK"},
    }
    mock_readiness.side_effect = lambda key, record: {"ready": key != "transfer"}
    mock_severity.return_value = {"prediction": {"prediction": "HIGH", "prediction_id": "p-sev"}}

    def stored(supabase, **kwargs):
        for key, output in kwargs["predictions"].items():
            output["prediction_id"] = f"p-{key}"
    mock_store.side_effect = lambda supabase, **kwargs: None    # the database write failed

    mock_accident_table = MagicMock()
    mock_accident_table.select.return_value.eq.return_value.eq.return_value.order.return_value.execute.return_value = MockDBResult(data=[
        {"accident_id": "acc-1", "hospital_id": HOSPITAL_A, "patient_id": "00000000-0000-0000-0000-000000000001"}
    ])

    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "doctor-id-123", "role": "doctor", "hospital_id": HOSPITAL_A}
            ])
        elif table_name == "Accident_Record":
            return mock_accident_table
        elif table_name == "Patient":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"patient_id": "00000000-0000-0000-0000-000000000001"}
            ])
        return mock_table

    mock_supabase.table.side_effect = table_side_effect

    response = client.get(
        "/api/patients/00000000-0000-0000-0000-000000000001/prognosis",
        headers={"Authorization": f"Bearer {doctor_token}"}
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "could not be saved" in response.json()["detail"]


def test_get_patient_prognosis_no_accident_record(client, mock_supabase, doctor_token):
    mock_accident_table = MagicMock()
    # Return empty list simulating no accident record found
    mock_accident_table.select.return_value.eq.return_value.eq.return_value.order.return_value.execute.return_value = MockDBResult(data=[])
    
    def table_side_effect(table_name):
        mock_table = MagicMock()
        if table_name == "User":
            mock_table.select.return_value.eq.return_value.execute.return_value = MockDBResult(data=[
                {"user_id": "doctor-id-123", "role": "doctor", "hospital_id": HOSPITAL_A}
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
    assert response.json()["detail"] == "Patient not found."
