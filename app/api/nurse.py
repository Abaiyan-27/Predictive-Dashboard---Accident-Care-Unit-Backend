"""Nurse router — patient intake and personal patient list."""
import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_nurse
from app.db.session import get_db
from app.models.patient import Patient
from app.models.user import User
from app.schemas.patient import PatientIntakeForm, PatientOut
from app.services import ml_service, audit_service

router = APIRouter(prefix="/patients", tags=["Nurse / Patients"])


@router.post("", response_model=PatientOut, status_code=status.HTTP_201_CREATED)
def create_patient(
    form: PatientIntakeForm,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_nurse),
):
    """
    Nurse submits patient intake form.
    ML service predicts severity, priority, and LOS.
    """
    prediction = ml_service.predict_severity(
        age=form.age,
        gender=form.gender,
        collision_type=form.collision_type,
        injury_description=form.injury_description,
    )

    patient = Patient(
        id=str(uuid.uuid4()),
        name=form.name,
        age=form.age,
        gender=form.gender,
        admission_date=date.today().isoformat(),
        collision_type=form.collision_type,
        injury_description=form.injury_description,
        severity=prediction.severity,
        priority=prediction.priority,
        predicted_los_days=prediction.predicted_los_days,
        created_by=current_user.id,
        hospital_id=current_user.hospital_id,
    )
    db.add(patient)

    audit_service.log_action(
        db,
        user_name=current_user.name,
        role=current_user.role,
        action=(
            f"Patient '{form.name}' admitted. "
            f"Severity: {prediction.severity}, Priority: {prediction.priority}, "
            f"LOS: {prediction.predicted_los_days} days."
        ),
    )

    db.commit()
    db.refresh(patient)
    return PatientOut.model_validate(patient)


@router.get("/my", response_model=list[PatientOut])
def get_my_patients(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return patients created by the authenticated nurse."""
    patients = (
        db.query(Patient).filter(Patient.created_by == current_user.id).all()
    )
    return [PatientOut.model_validate(p) for p in patients]
