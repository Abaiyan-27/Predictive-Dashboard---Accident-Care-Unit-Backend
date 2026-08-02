"""Doctor router — hospital patients, patient detail, and prognosis."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import (
    get_current_user,
    require_doctor,
    require_doctor_or_admin,
    require_nurse_doctor_admin,
)
from app.db.session import get_db
from app.models.patient import Patient
from app.models.user import User
from app.schemas.patient import PatientOut, PrognosisOut
from app.services import ml_service

router = APIRouter(tags=["Doctor / Patients"])


@router.get("/patients/hospital/{hospital_id}", response_model=list[PatientOut])
def get_hospital_patients(
    hospital_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor_or_admin),
):
    """Return all patients admitted to a specific hospital."""
    patients = db.query(Patient).filter(Patient.hospital_id == hospital_id).all()
    return [PatientOut.model_validate(p) for p in patients]


@router.get("/patients/{patient_id}", response_model=PatientOut)
def get_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_nurse_doctor_admin),
):
    """Return a single patient record."""
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found.")
    return PatientOut.model_validate(patient)


@router.get("/patients/{patient_id}/prognosis", response_model=PrognosisOut)
def get_patient_prognosis(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor),
):
    """Run ML prognosis model for a patient."""
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found.")

    prediction = ml_service.predict_prognosis(
        severity=patient.severity,
        age=patient.age,
        collision_type=patient.collision_type,
        predicted_los_days=patient.predicted_los_days,
    )

    return PrognosisOut(
        expected_recovery_level=prediction.expected_recovery_level,
        partial_recovery_probability=prediction.partial_recovery_probability,
        transfer_recommendation=prediction.transfer_recommendation,
    )
