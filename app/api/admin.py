"""Admin router — hospital info, user management, transfers, audit logs.

Admins belong to one hospital and every endpoint here is limited to it.
There is no system-wide admin role.
"""
import secrets
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from postgrest.exceptions import APIError
from supabase import Client

from app.api.deps import Page, page_params, require_admin, require_doctor_or_admin
from app.core.authorization import (
    Principal,
    exact_ilike,
    ensure_own_hospital_id,
    load_accident,
    load_transfer,
)
from app.db.session import get_supabase
from app.schemas.hospital import HospitalOut
from app.schemas.user import CreateUserPayload, CreatedUserOut, UserOut
from app.schemas.transfer import (
    AuthorizeTransferPayload,
    CreateTransferPayload,
    RejectTransferPayload,
    TransferOut,
)
from app.schemas.audit_log import AuditLogOut
from app.services import audit_service, doctor_workflow, patient_service, transfer_service

router = APIRouter(prefix="/admin", tags=["Admin"])


# ── Hospital Info ────────────────────────────────────────────────────────────

@router.get("/hospitals", response_model=list[HospitalOut])
def get_hospitals(
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_admin),
):
    """Return the admin's own hospital (the only one they administer)."""
    result = supabase.table("Hospital").select("*").eq("hospital_id", current_user.hospital_id).execute()
    return result.data or []


@router.get("/hospitals/{hospital_id}", response_model=HospitalOut)
def get_hospital(
    hospital_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_admin),
):
    """Return the admin's own hospital."""
    own_hospital = ensure_own_hospital_id(current_user, hospital_id)
    result = supabase.table("Hospital").select("*").eq("hospital_id", own_hospital).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Hospital not found.")
    return result.data[0]


# ── User Management ─────────────────────────────────────────────────────────

@router.get("/users", response_model=list[UserOut])
def get_users(
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_admin),
):
    """Return the staff of the admin's own hospital."""
    result = supabase.table("User").select("*").eq("hospital_id", current_user.hospital_id).execute()
    return result.data or []


def _find_auth_user(supabase: Client, email: str):
    """The Supabase Auth user with this email (case-insensitive), searching every page."""
    wanted = email.strip().lower()
    page = 1
    while page <= 100:
        users = supabase.auth.admin.list_users(page=page, per_page=1000) or []
        for user in users:
            if (getattr(user, "email", None) or "").lower() == wanted:
                return user
        if len(users) < 1000:
            return None
        page += 1
    return None


def _is_unused_setup_account(auth_user) -> bool:
    """Created by this endpoint (still flagged) and never signed in to."""
    metadata = getattr(auth_user, "user_metadata", None)
    flagged = isinstance(metadata, dict) and metadata.get("must_change_password") is True
    return flagged and not getattr(auth_user, "last_sign_in_at", None)


@router.post("/users", response_model=CreatedUserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: CreateUserPayload,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_admin),
):
    """Create a staff login (nurse, doctor or admin) for the admin's own hospital.

    Without a password in the request a temporary one is generated and returned
    once, for the admin to hand over. Either way the account is flagged so the
    new user must replace the password at first sign-in (/auth/change-password).
    No email is sent: the backend has no mail service configured.
    """
    own_hospital = ensure_own_hospital_id(current_user, payload.hospital_id)
    temporary_password = None if payload.password else secrets.token_urlsafe(12)
    password = payload.password or temporary_password

    email_pattern = exact_ilike(payload.email)
    existing = supabase.table("User").select("user_id").ilike("email", email_pattern).execute()
    officer = supabase.table("Government_Officer").select("officer_id").ilike("email", email_pattern).execute()
    if existing.data or officer.data:
        # Also stops the auth-account recovery below from resetting the
        # password of an existing staff or government login.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with that email already exists.",
        )

    created_auth_account = False
    try:
        auth_result = supabase.auth.admin.create_user({
            "email": payload.email,
            "password": password,
            "email_confirm": True,
            "user_metadata": {"must_change_password": True},
        })
        auth_user = auth_result.user
        created_auth_account = True
    except Exception as exc:
        # Recover an Auth-only account left by an interrupted earlier setup:
        # one this endpoint created (first-sign-in flag still set) that has
        # never been used. Any other Auth account with this email belongs to
        # someone; resetting its password here would hand it to the admin.
        auth_user = _find_auth_user(supabase, payload.email)
        if not auth_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Unable to create the authentication account.",
            ) from exc
        if not _is_unused_setup_account(auth_user):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "An authentication account with this email already exists and has been used. "
                    "It cannot be reassigned here; ask the system administrator to review it."
                ),
            ) from exc
        supabase.auth.admin.update_user_by_id(auth_user.id, {
            "password": password,
            "email_confirm": True,
            "user_metadata": {"must_change_password": True},
        })

    if not auth_user:
        raise HTTPException(status_code=500, detail="Authentication account was not created.")

    user_data = payload.model_dump(exclude={"password"}, exclude_none=True)
    user_data["user_id"] = auth_user.id
    user_data["hospital_id"] = own_hospital

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

    created = result.data[0]
    audit_service.log_event(
        supabase,
        actor_id=current_user.id,
        role=current_user.role,
        action_type="user_created",
        hospital_id=current_user.hospital_id,
        details={"user_id": created.get("user_id"), "name": created.get("name"),
                 "email": created.get("email"), "role": created.get("role")},
    )
    return {**created, "temporary_password": temporary_password, "must_change_password": True}


# ── Patients (read-only operational view) ────────────────────────────────────

ADMIN_MODELS = ("stay_duration", "transfer")


def _prediction_view(model_status: dict) -> dict:
    """What an admin sees of a model: its current prediction, never a decision control."""
    prediction = model_status.get("prediction")
    return {
        "title": model_status["title"],
        "state": model_status["state"],
        "prediction": None if not prediction else {
            "prediction_id": prediction.get("prediction_id"),
            "prediction": prediction.get("prediction"),
            "status": prediction.get("status"),
            "probability": prediction.get("probability"),
            "predicted_at": prediction.get("predicted_at"),
            "stale": prediction.get("stale"),
            "stale_reason": prediction.get("stale_reason"),
        },
    }


def _patient_basics(accident: dict) -> dict:
    patient = patient_service._first(accident.get("Patient"))
    return {key: patient.get(key) for key in ("patient_id", "full_name", "age", "gender")}


@router.get("/patients")
def get_admin_patients(
    response: Response,
    page: Page = Depends(page_params),
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_admin),
):
    """The admin's hospital episodes: basics, Stay and Transfer predictions, open transfer."""
    result = page.apply(
        supabase.table("Accident_Record")
        .select(doctor_workflow.ACCIDENT_SELECT, count="exact")
        .eq("hospital_id", current_user.hospital_id)
        .order("created_at", desc=True)
    ).execute()
    Page.report_total(response, result.count)
    accidents = result.data or []
    predictions, decisions = doctor_workflow.load_history(supabase, [a["accident_id"] for a in accidents])
    rows = []
    for accident in accidents:
        models = doctor_workflow.models_status(
            accident, predictions.get(str(accident["accident_id"]), {}), decisions, role=current_user.role
        )
        open_transfer = next(
            (t for t in patient_service._list(accident.get("Transfer")) if t.get("status") in transfer_service.OPEN_STATUSES),
            None,
        )
        rows.append({
            "accident_id": accident["accident_id"],
            "created_at": accident.get("created_at"),
            "status": accident.get("status"),
            "patient": _patient_basics(accident),
            "stay_duration": _prediction_view(models["stay_duration"]),
            "transfer": _prediction_view(models["transfer"]),
            "open_transfer": None if not open_transfer else {
                "transfer_id": open_transfer["transfer_id"],
                "status": open_transfer.get("status"),
                "to_hospital_id": open_transfer.get("to_hospital_id"),
            },
        })
    return rows


@router.get("/episodes/{accident_id}")
def get_admin_episode(
    accident_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_admin),
):
    """One episode as the admin sees it: basics, stays, Stay/Transfer predictions, transfers."""
    accident = load_accident(supabase, current_user, accident_id, select=doctor_workflow.ACCIDENT_SELECT)
    models = doctor_workflow.load_models_status(supabase, accident, role=current_user.role)
    transfers = transfer_service.enrich(
        supabase,
        sorted(patient_service._list(accident.get("Transfer")),
               key=lambda t: str(t.get("requested_at") or ""), reverse=True),
        current_user,
    )
    stays = (
        supabase.table("Hospital_Patient").select("id,hospital_id,entered_at,exited_at")
        .eq("accident_id", str(accident_id)).order("entered_at").execute().data or []
    )
    hospital_names = transfer_service._names(supabase, "Hospital", "hospital_id",
                                             [s.get("hospital_id") for s in stays] + [accident.get("hospital_id")])
    for stay in stays:
        stay["hospital_name"] = hospital_names.get(str(stay.get("hospital_id")))
    open_transfer = next((t for t in transfers if t.get("status") in transfer_service.OPEN_STATUSES), None)
    return {
        "accident_id": accident["accident_id"],
        "patient": _patient_basics(accident),
        "episode": {
            "created_at": accident.get("created_at"),
            "status": accident.get("status"),
            "hospital_id": accident.get("hospital_id"),
            "hospital_name": hospital_names.get(str(accident.get("hospital_id"))),
            "incident_at": accident.get("incident_at"),
            "injuries": [
                {"site_of_injury": i.get("site_of_injury"), "type_of_injury": i.get("type_of_injury")}
                for i in patient_service.order_injuries(accident.get("Injury_Description"))
            ],
        },
        "stays": stays,
        "predictions": {key: _prediction_view(models[key]) for key in ADMIN_MODELS},
        "transfers": transfers,
        "open_transfer": open_transfer,
        "can_request_transfer": open_transfer is None,
    }


# ── Transfers ────────────────────────────────────────────────────────────────

@router.get("/network-hospitals")
def get_network_hospitals(
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_admin),
):
    """The hospitals a transfer may go to (the network), excluding the admin's own."""
    return [h for h in transfer_service.network_hospitals(supabase)
            if str(h["hospital_id"]) != current_user.hospital_id]


@router.get("/transfers", response_model=list[TransferOut])
def get_transfers(
    response: Response,
    page: Page = Depends(page_params),
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_doctor_or_admin),
):
    """Transfers that start or end at the caller's hospital, most recent first (paged)."""
    own = current_user.hospital_id
    result = page.apply(
        supabase.table("Transfer")
        .select("*", count="exact")
        .or_(f"from_hospital_id.eq.{own},to_hospital_id.eq.{own}")
        .order("requested_at", desc=True)
    ).execute()
    Page.report_total(response, result.count)
    return transfer_service.enrich(supabase, result.data or [], current_user)


@router.post("/transfers", response_model=TransferOut, status_code=status.HTTP_201_CREATED)
def create_transfer(
    payload: CreateTransferPayload,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_admin),
):
    """Request a transfer of an episode held by the admin's hospital to a network hospital.

    The request appears on the sending hospital's nurse dashboard. One open
    request per episode; it is an operational request, not a model decision.
    """
    own_hospital = ensure_own_hospital_id(current_user, payload.from_hospital_id, "from_hospital_id")
    load_accident(supabase, current_user, payload.accident_id, select="accident_id,hospital_id")
    destination = str(payload.to_hospital_id)
    if destination == own_hospital:
        raise HTTPException(status_code=422, detail="A transfer must go to a different hospital.")
    if destination not in {str(h["hospital_id"]) for h in transfer_service.network_hospitals(supabase)}:
        raise HTTPException(status_code=422, detail="The destination must be one of the network hospitals.")
    open_requests = (
        supabase.table("Transfer").select("transfer_id")
        .eq("accident_id", str(payload.accident_id))
        .in_("status", list(transfer_service.OPEN_STATUSES))
        .execute().data
    )
    if open_requests:
        raise HTTPException(status_code=409, detail="This patient already has an open transfer request.")
    try:
        result = supabase.table("Transfer").insert({
            "accident_id": str(payload.accident_id),
            "from_hospital_id": own_hospital,
            "to_hospital_id": destination,
            "request_message": payload.message.strip(),
            "requested_by": current_user.id,
            "status": transfer_service.PENDING,
        }).execute()
    except APIError as exc:
        if str(exc.code) in {"23505"}:
            raise HTTPException(status_code=409, detail="This patient already has an open transfer request.") from exc
        if str(exc.code) == "PT422":
            raise HTTPException(status_code=422, detail=exc.message) from exc
        raise
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create transfer.")
    transfer = transfer_service.enrich(supabase, [result.data[0]], current_user)[0]
    audit_service.log_event(
        supabase,
        actor_id=current_user.id,
        role=current_user.role,
        action_type="transfer_requested",
        hospital_id=own_hospital,
        accident_id=payload.accident_id,
        details=transfer_service.audit_details(transfer, old_status=None, new_status=transfer["status"]),
    )
    return transfer


@router.post("/transfers/{transfer_id}/authorize", response_model=TransferOut)
def authorize_transfer(
    transfer_id: UUID,
    body: AuthorizeTransferPayload,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_admin),
):
    """Sending-hospital admin authorizes a pending request; the nurses may then process it."""
    transfer = transfer_service.enrich(supabase, [load_transfer(supabase, current_user, transfer_id)])[0]
    return transfer_service.transition(supabase, current_user, transfer, "authorize", note=(body.note or "").strip() or None)


@router.post("/transfers/{transfer_id}/reject", response_model=TransferOut)
def reject_transfer(
    transfer_id: UUID,
    body: RejectTransferPayload,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_admin),
):
    """Sending-hospital admin rejects (cancels) a pending or authorized request."""
    transfer = transfer_service.enrich(supabase, [load_transfer(supabase, current_user, transfer_id)])[0]
    return transfer_service.transition(supabase, current_user, transfer, "reject", note=body.reason.strip())


# ── Audit Logs ───────────────────────────────────────────────────────────────

@router.get("/audit-logs", response_model=list[AuditLogOut])
def get_audit_logs(
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_admin),
    action: Optional[str] = Query(default=None, max_length=100, description="Action type, or a prefix ending in *"),
    accident_id: Optional[UUID] = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """The admin's hospital audit log, newest first, with actor, hospital and patient names."""
    query = supabase.table("AuditLog").select("*").eq("hospital_id", current_user.hospital_id)
    if action:
        query = query.like("action_type", action[:-1] + "%") if action.endswith("*") else query.eq("action_type", action)
    if accident_id:
        query = query.eq("accident_id", str(accident_id))
    rows = query.order("timestamp", desc=True).range(offset, offset + limit - 1).execute().data or []

    actors = transfer_service._names(supabase, "User", "user_id", [r.get("actor_id") for r in rows])
    missing = [r.get("actor_id") for r in rows if str(r.get("actor_id")) not in actors]
    actors.update(transfer_service._names(supabase, "Government_Officer", "officer_id", missing))
    hospitals = transfer_service._names(supabase, "Hospital", "hospital_id", [r.get("hospital_id") for r in rows])
    accident_ids = sorted({str(r["accident_id"]) for r in rows if r.get("accident_id")})
    patients: dict[str, str] = {}
    if accident_ids:
        for row in (supabase.table("Accident_Record").select("accident_id,Patient(full_name)")
                    .in_("accident_id", accident_ids).execute().data or []):
            patients[str(row["accident_id"])] = patient_service._first(row.get("Patient")).get("full_name")
    for row in rows:
        row["actor_name"] = actors.get(str(row.get("actor_id")))
        row["hospital_name"] = hospitals.get(str(row.get("hospital_id")))
        row["patient_name"] = patients.get(str(row.get("accident_id")))
    return rows
