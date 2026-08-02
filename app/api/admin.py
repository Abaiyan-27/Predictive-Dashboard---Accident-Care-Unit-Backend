"""Admin router — hospital status, user management, transfers, audit logs."""
import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import (
    get_current_user,
    require_admin,
    require_doctor_or_admin,
)
from app.core.security import get_password_hash
from app.db.session import get_db
from app.models.audit_log import AuditLog
from app.models.hospital import Hospital
from app.models.patient import Patient
from app.models.transfer import Transfer
from app.models.user import User
from app.schemas.audit_log import AuditLogOut
from app.schemas.hospital import HospitalStatus
from app.schemas.patient import PatientOut
from app.schemas.transfer import TransferOut, UpdateTransferStatus
from app.schemas.user import CreateUserPayload, UserOut
from app.services import audit_service

router = APIRouter(prefix="/admin", tags=["Admin"])


# ── Hospital Status ──────────────────────────────────────────────────────────

@router.get("/hospital-status", response_model=HospitalStatus)
def get_hospital_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    hospital = db.get(Hospital, current_user.hospital_id)
    if not hospital:
        raise HTTPException(status_code=404, detail="Hospital record not found.")

    # Sync occupied_beds with actual patient count in real time
    patient_count = (
        db.query(Patient).filter(Patient.hospital_id == current_user.hospital_id).count()
    )
    occupied = min(patient_count, hospital.total_beds)
    hospital.occupied_beds = occupied
    hospital.available_beds = hospital.total_beds - occupied
    db.commit()

    return HospitalStatus(
        total_beds=hospital.total_beds,
        occupied_beds=hospital.occupied_beds,
        available_beds=hospital.available_beds,
    )


# ── User Management ──────────────────────────────────────────────────────────

@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: CreateUserPayload,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with that email already exists.",
        )

    new_user = User(
        id=str(uuid.uuid4()),
        name=payload.name,
        email=payload.email,
        hashed_password=get_password_hash("demo"),
        role=payload.role,
        hospital_id=payload.hospital_id,
        hospital_name=payload.hospital_name,
    )
    db.add(new_user)

    audit_service.log_action(
        db,
        user_name=current_user.name,
        role=current_user.role,
        action=f"Created new user '{payload.name}' ({payload.role}) at hospital '{payload.hospital_id}'.",
    )

    db.commit()
    db.refresh(new_user)
    return UserOut.model_validate(new_user)


# ── Transfers ────────────────────────────────────────────────────────────────

@router.get("/transfers", response_model=list[TransferOut])
def get_transfers(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor_or_admin),
):
    transfers = db.query(Transfer).order_by(Transfer.requested_date.desc()).all()
    return [TransferOut.model_validate(t) for t in transfers]


@router.patch("/transfers/{transfer_id}/status", response_model=TransferOut)
def update_transfer_status(
    transfer_id: str,
    body: UpdateTransferStatus,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    transfer = db.get(Transfer, transfer_id)
    if not transfer:
        raise HTTPException(status_code=404, detail="Transfer request not found.")

    transfer.status = body.status

    audit_service.log_action(
        db,
        user_name=current_user.name,
        role=current_user.role,
        action=(
            f"Transfer for '{transfer.patient_name}' "
            f"({transfer.source_hospital} → {transfer.destination_hospital}) "
            f"marked as '{body.status}'."
        ),
    )

    db.commit()
    db.refresh(transfer)
    return TransferOut.model_validate(transfer)


# ── Audit Logs ───────────────────────────────────────────────────────────────

@router.get("/audit-logs", response_model=list[AuditLogOut])
def get_audit_logs(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    logs = (
        db.query(AuditLog)
        .order_by(AuditLog.timestamp.desc())
        .all()
    )
    return [AuditLogOut.model_validate(log) for log in logs]
