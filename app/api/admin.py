"""Admin router — hospital info, user management, transfers, audit logs."""
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from app.api.deps import (
    get_current_user,
    require_admin,
    require_doctor_or_admin,
)
from app.db.session import get_supabase
from app.schemas.hospital import HospitalOut, CreateHospitalPayload
from app.schemas.user import CreateUserPayload, UserOut
from app.schemas.transfer import TransferOut, CreateTransferPayload, UpdateTransferStatus
from app.schemas.audit_log import AuditLogOut
from app.services import audit_service

router = APIRouter(prefix="/admin", tags=["Admin"])


# ── Hospital Info ────────────────────────────────────────────────────────────

@router.get("/hospitals", response_model=list[HospitalOut])
def get_hospitals(
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(require_admin),
):
    """Return all hospitals."""
    result = supabase.table("Hospital").select("*").execute()
    return result.data or []


@router.get("/hospitals/{hospital_id}", response_model=HospitalOut)
def get_hospital(
    hospital_id: str,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(require_admin),
):
    """Return a specific hospital."""
    result = supabase.table("Hospital").select("*").eq("hospital_id", hospital_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Hospital not found.")
    return result.data[0]


@router.post("/hospitals", response_model=HospitalOut, status_code=status.HTTP_201_CREATED)
def create_hospital(
    payload: CreateHospitalPayload,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(require_admin),
):
    """Create a new hospital."""
    hospital_data = payload.model_dump(exclude_none=True)
    result = supabase.table("Hospital").insert(hospital_data).execute()
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create hospital.")

    audit_service.log_action(
        supabase,
        actor_id=current_user.get("user_id"),
        role=current_user.get("role"),
        action_type="hospital_created",
        hospital_id=result.data[0]["hospital_id"],
    )
    return result.data[0]


# ── User Management ─────────────────────────────────────────────────────────

@router.get("/users", response_model=list[UserOut])
def get_users(
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(require_admin),
):
    """Return all users."""
    result = supabase.table("User").select("*").execute()
    return result.data or []


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: CreateUserPayload,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(require_admin),
):
    """Create a staff login in Supabase Auth and its matching public profile."""
    # Check for existing user
    existing = supabase.table("User").select("user_id").eq("email", payload.email).execute()
    if existing.data:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with that email already exists.",
        )

    created_auth_account = False
    try:
        auth_result = supabase.auth.admin.create_user({
            "email": payload.email,
            "password": payload.password,
            "email_confirm": True,
        })
        auth_user = auth_result.user
        created_auth_account = True
    except Exception as exc:
        # Recover an Auth-only account left by an interrupted earlier setup.
        auth_user = next(
            (
                user for user in supabase.auth.admin.list_users(per_page=1000)
                if user.email == payload.email
            ),
            None,
        )
        if not auth_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Unable to create the authentication account.",
            ) from exc
        supabase.auth.admin.update_user_by_id(auth_user.id, {
            "password": payload.password,
            "email_confirm": True,
        })

    if not auth_user:
        raise HTTPException(status_code=500, detail="Authentication account was not created.")

    user_data = payload.model_dump(exclude={"password"}, exclude_none=True)
    user_data["user_id"] = auth_user.id
    # Convert UUID fields to strings for Supabase
    if "hospital_id" in user_data and user_data["hospital_id"]:
        user_data["hospital_id"] = str(user_data["hospital_id"])

    try:
        result = supabase.table("User").insert(user_data).execute()
    except Exception as exc:
        # Avoid leaving an unusable Auth-only account when profile creation fails.
        try:
            if created_auth_account:
                supabase.auth.admin.delete_user(auth_user.id)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Failed to create user profile.") from exc
    if not result.data:
        try:
            if created_auth_account:
                supabase.auth.admin.delete_user(auth_user.id)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Failed to create user.")

    audit_service.log_action(
        supabase,
        actor_id=current_user.get("user_id"),
        role=current_user.get("role"),
        action_type="user_created",
        hospital_id=current_user.get("hospital_id"),
    )

    return result.data[0]


# ── Transfers ────────────────────────────────────────────────────────────────

@router.get("/transfers", response_model=list[TransferOut])
def get_transfers(
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(require_doctor_or_admin),
):
    """Return all transfer requests, ordered by most recent."""
    result = (
        supabase.table("Transfer")
        .select("*")
        .order("requested_at", desc=True)
        .execute()
    )
    transfers = result.data or []
    for transfer in transfers:
        accident_id = transfer.get("accident_id")
        if accident_id:
            accident_result = (
                supabase.table("Accident_Record")
                .select("Patient(full_name)")
                .eq("accident_id", accident_id)
                .limit(1)
                .execute()
            )
            if accident_result.data:
                patient = accident_result.data[0].get("Patient")
                if isinstance(patient, list):
                    patient = patient[0] if patient else None
                if patient:
                    transfer["patient_name"] = patient.get("full_name")

        for id_field, name_field in (
            ("from_hospital_id", "source_hospital"),
            ("to_hospital_id", "destination_hospital"),
        ):
            hospital_id = transfer.get(id_field)
            if not hospital_id:
                continue
            hospital_result = (
                supabase.table("Hospital")
                .select("name")
                .eq("hospital_id", hospital_id)
                .limit(1)
                .execute()
            )
            if hospital_result.data:
                transfer[name_field] = hospital_result.data[0].get("name")

    return transfers


@router.post("/transfers", response_model=TransferOut, status_code=status.HTTP_201_CREATED)
def create_transfer(
    payload: CreateTransferPayload,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(require_doctor_or_admin),
):
    """Create a new transfer request."""
    transfer_data = {
        "accident_id": str(payload.accident_id),
        "from_hospital_id": str(payload.from_hospital_id),
        "to_hospital_id": str(payload.to_hospital_id),
        "authorization_reason": payload.authorization_reason,
        "state_of_transfer": payload.state_of_transfer,
        "requested_by": current_user.get("user_id"),
        "status": "pending_authorization",
    }
    result = supabase.table("Transfer").insert(transfer_data).execute()
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create transfer.")

    audit_service.log_action(
        supabase,
        actor_id=current_user.get("user_id"),
        role=current_user.get("role"),
        action_type="transfer_requested",
        hospital_id=current_user.get("hospital_id"),
        accident_id=payload.accident_id,
    )
    return result.data[0]


@router.patch("/transfers/{transfer_id}/status", response_model=TransferOut)
def update_transfer_status(
    transfer_id: str,
    body: UpdateTransferStatus,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(require_admin),
):
    """Update a transfer request status (authorize, approve, reject)."""
    # Get existing transfer
    existing = supabase.table("Transfer").select("*").eq("transfer_id", transfer_id).execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Transfer request not found.")

    update_data = {"status": body.status}

    if body.status == "authorized":
        update_data["authorized_by"] = current_user.get("user_id")
    elif body.status in ("approved", "rejected"):
        update_data["approved_by"] = current_user.get("user_id")

    result = (
        supabase.table("Transfer")
        .update(update_data)
        .eq("transfer_id", transfer_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to update transfer.")

    # If approved, update accident record's hospital_id and create Hospital_Patient record
    transfer = result.data[0]
    if body.status == "approved":
        accident_id = transfer.get("accident_id")
        to_hospital = transfer.get("to_hospital_id")
        from_hospital = transfer.get("from_hospital_id")

        if accident_id and to_hospital:
            # Update accident record hospital
            supabase.table("Accident_Record").update(
                {"hospital_id": to_hospital}
            ).eq("accident_id", accident_id).execute()

            # Close old Hospital_Patient record
            supabase.table("Hospital_Patient").update(
                {"exited_at": "now()"}
            ).eq("accident_id", accident_id).eq("hospital_id", from_hospital).is_("exited_at", "null").execute()

            # Create new Hospital_Patient record
            supabase.table("Hospital_Patient").insert({
                "accident_id": accident_id,
                "hospital_id": to_hospital,
            }).execute()

    audit_service.log_action(
        supabase,
        actor_id=current_user.get("user_id"),
        role=current_user.get("role"),
        action_type=f"transfer_{body.status}",
        hospital_id=current_user.get("hospital_id"),
        accident_id=transfer.get("accident_id"),
    )

    return transfer


# ── Audit Logs ───────────────────────────────────────────────────────────────

@router.get("/audit-logs", response_model=list[AuditLogOut])
def get_audit_logs(
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(require_admin),
):
    """Return all audit logs, ordered by most recent."""
    result = (
        supabase.table("AuditLog")
        .select("*")
        .order("timestamp", desc=True)
        .execute()
    )
    return result.data or []
