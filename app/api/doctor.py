"""Doctor router — hospital patients, patient detail, and prognosis."""
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from app.api.deps import (
    get_current_user,
    require_doctor,
    require_doctor_or_admin,
    require_nurse_doctor_admin,
)
from app.db.session import get_supabase
from app.schemas.accident import TreatmentCreatePayload, TreatmentOut
from app.services import ml_service, audit_service, patient_service

router = APIRouter(tags=["Doctor / Patients"])


@router.get("/patients/hospital/{hospital_id}")
def get_hospital_patients(
    hospital_id: str,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(require_doctor_or_admin),
):
    """Return all accident records for a specific hospital, with patient and injury details."""
    if current_user.get("role") == "doctor" and str(current_user.get("hospital_id")) != hospital_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Doctors can only view patients assigned to their hospital.",
        )

    result = (
        supabase.table("Accident_Record")
        .select("*, Patient(*), Injury_Description(*), Treatment(*), Transfer(*)")
        .eq("hospital_id", hospital_id)
        .order("created_at", desc=True)
        .execute()
    )
    return patient_service.add_severity_predictions(result.data or [])


@router.get("/patients/{patient_id}")
def get_patient(
    patient_id: str,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(require_nurse_doctor_admin),
):
    """Return a single patient record with their accident records and injuries."""
    patient_result = supabase.table("Patient").select("*").eq("patient_id", patient_id).execute()
    if not patient_result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found.")

    patient = patient_result.data[0]

    # Fetch related accident records with injuries
    accidents_result = (
        supabase.table("Accident_Record")
        .select("*, Patient(*), Injury_Description(*), Treatment(*), Transfer(*)")
        .eq("patient_id", patient_id)
        .execute()
    )
    patient["accident_records"] = patient_service.add_severity_predictions(
        accidents_result.data or []
    )

    # Fetch socioeconomic info
    socio_result = (
        supabase.table("Socioeconomic_Info")
        .select("*")
        .eq("patient_id", patient_id)
        .execute()
    )
    patient["socioeconomic_info"] = socio_result.data[0] if socio_result.data else None

    return patient


@router.get("/accidents/{accident_id}")
def get_accident_detail(
    accident_id: str,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(require_nurse_doctor_admin),
):
    """Return full accident record with patient, injuries, and treatments."""
    result = (
        supabase.table("Accident_Record")
        .select("*, Patient(*), Injury_Description(*), Treatment(*), Transfer(*)")
        .eq("accident_id", accident_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Accident record not found.")
    return patient_service.add_severity_prediction(result.data[0])


@router.post("/treatments", status_code=status.HTTP_201_CREATED)
def create_treatment(
    payload: TreatmentCreatePayload,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(require_doctor),
):
    """Doctor logs a treatment for an accident record."""
    treatment_data = {
        "accident_id": str(payload.accident_id) if payload.accident_id else None,
        "hospital_id": str(payload.hospital_id) if payload.hospital_id else None,
        "ward_number": payload.ward_number,
        "reason": payload.reason,
        "number_of_days_stay": payload.number_of_days_stay,
        "investigation_done": payload.investigation_done,
        "management_name_1": payload.management_name_1,
        "management_name_2": payload.management_name_2,
        "decided_by": str(payload.decided_by) if payload.decided_by else current_user.get("user_id"),
        "logged_by": current_user.get("user_id"),
    }
    result = supabase.table("Treatment").insert(treatment_data).execute()
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create treatment.")

    # Audit log
    audit_service.log_action(
        supabase,
        actor_id=current_user.get("user_id"),
        role=current_user.get("role"),
        action_type="treatment_logged",
        hospital_id=payload.hospital_id,
        accident_id=payload.accident_id,
    )

    return result.data[0]


@router.get("/patients/{patient_id}/prognosis")
def get_patient_prognosis(
    patient_id: str,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(require_doctor),
):
    """Run ML prognosis model for a patient based on their accident record."""
    # Get patient's latest accident record
    accidents = (
        supabase.table("Accident_Record")
        .select("*, Injury_Description(*), Treatment(*)")
        .eq("patient_id", patient_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not accidents.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No accident record found for patient.")

    accident = accidents.data[0]
    patient_result = supabase.table("Patient").select("*").eq("patient_id", patient_id).execute()
    patient = patient_result.data[0] if patient_result.data else {}

    transfer_result = (
        supabase.table("Transfer")
        .select("*")
        .eq("accident_id", accident.get("accident_id"))
        .order("requested_at", desc=True)
        .execute()
    )
    model_record = patient_service.build_discharge_record(
        patient=patient,
        accident=accident,
        transfers=transfer_result.data or [],
    )
    severity = ml_service.predict_severity(record=model_record)
    stay_duration = ml_service.predict_stay_duration(model_record)
    transfer = ml_service.predict_transfer_probability(model_record)
    discharge = ml_service.predict_discharge_outcome(model_record)
    return {
        "severity": severity.to_dict(),
        "stay_duration": stay_duration.to_dict(),
        "transfer": transfer.to_dict(),
        "discharge": discharge.to_dict(),
    }
