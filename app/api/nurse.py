"""Nurse router — patient intake and the hospital's patient list.

Saving intake details never runs a model. The nurse runs Severity explicitly
(app/api/severity.py); nurses never run the stay, transfer or discharge models.
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from postgrest.exceptions import APIError
from supabase import Client

from app.api.deps import Page, page_params, require_nurse, require_staff
from app.core.authorization import (
    NURSE,
    Principal,
    ensure_own_hospital_id,
    filter_predictions,
    load_accident,
    load_transfer,
)
from app.db.session import get_supabase
from app.schemas.accident import FullIntakeForm, UpdateIntakeForm
from app.schemas.transfer import ProcessTransferPayload, TransferOut
from app.services import audit_service, patient_service, schema_service, severity_workflow, transfer_service
from app.services.severity_workflow import ACCIDENT_SELECT, EMBEDDED_KEYS

router = APIRouter(prefix="/patients", tags=["Nurse / Patients"])


@router.post("/intake", status_code=status.HTTP_201_CREATED)
def create_patient_intake(
    form: FullIntakeForm,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_nurse),
):
    """
    Nurse submits the intake form (basic details at minimum).
    Creates Patient, Accident_Record, Injury_Description(s) and Hospital_Patient
    records. No model is run; the response carries the Severity workflow state.
    """
    # Verify the expanded Supabase schema before creating Patient. This prevents
    # a failed Accident_Record insert from leaving an orphan patient behind.
    try:
        schema_service.require_intake_schema(supabase)
    except schema_service.DatabaseMigrationRequired as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    # The episode belongs to the nurse's own hospital, taken from their account;
    # a different hospital_id in the request is refused before anything is written.
    hospital_id = ensure_own_hospital_id(current_user, form.hospital_id)

    # 1. Create Patient
    patient_data = {
        "full_name": form.full_name,
        "age": form.age,
        "gender": form.gender,
    }
    patient_result = supabase.table("Patient").insert(patient_data).execute()
    if not patient_result.data:
        raise HTTPException(status_code=500, detail="Failed to create patient.")
    patient = patient_result.data[0]
    patient_id = patient["patient_id"]

    # 2. Create Accident Record (including ML feature columns)
    accident_data = {
        "patient_id": patient_id,
        "hospital_id": str(hospital_id),
        "first_hospital_id": str(form.first_hospital_id or hospital_id),
        "created_by": current_user.id,
        "vehicle_type": form.vehicle_type,
        "helmet_worn": form.helmet_worn,
        "road_type": form.road_type,
        "speed": form.speed,
        "incident_at": form.incident_at.isoformat() if form.incident_at else None,
        "road_condition": form.road_condition,
        "visibility": form.visibility,
        "road_signals_exist": form.road_signals_exist,
        "collision_force_from": form.collision_force_from,
        "alcohol_consumption": form.alcohol_consumption,
        "illicit_drugs": form.illicit_drugs,
        "status": "intake",
        # ML model feature columns
        "mode_of_transport": form.mode_of_transport,
        "time_to_hospital": form.time_to_hospital,
        "collision_with": form.collision_with,
        # Severity model raw inputs
        "mode_of_travel_during_accident": form.mode_of_travel_during_accident,
        "passenger_type": form.passenger_type,
        "other_injury": form.other_injury,
    }
    try:
        accident_result = supabase.table("Accident_Record").insert(accident_data).execute()
        if not accident_result.data:
            raise HTTPException(status_code=500, detail="Failed to create accident record.")
    except Exception:
        # Supabase's REST API does not span these two inserts in one transaction.
        # Remove only the patient created by this request when the second insert
        # fails, restoring the all-or-nothing behavior expected by intake.
        try:
            supabase.table("Patient").delete().eq("patient_id", patient_id).execute()
        except Exception:
            pass
        raise
    accident = accident_result.data[0]
    accident_id = accident["accident_id"]

    # 3. Create Injury Descriptions
    injuries_created = []
    if form.injuries:
        for injury in form.injuries:
            injury_data = {
                "accident_id": accident_id,
                "injury_order": injury.injury_order,
                "site_of_injury": injury.site_of_injury,
                "type_of_injury": injury.type_of_injury,
            }
            injury_result = supabase.table("Injury_Description").insert(injury_data).execute()
            if injury_result.data:
                injuries_created.append(injury_result.data[0])

    # 4. Create Hospital_Patient tracking record
    supabase.table("Hospital_Patient").insert({
        "accident_id": accident_id,
        "hospital_id": str(hospital_id),
    }).execute()

    # 5. Audit log
    audit_service.log_action(
        supabase,
        actor_id=current_user.id,
        role=current_user.role,
        action_type="patient_intake",
        hospital_id=hospital_id,
        accident_id=accident_id,
    )

    # 6. A new episode has no predictions yet; Severity runs when the nurse asks.
    severity = severity_workflow.episode_status(
        {**accident, "Patient": patient, "Injury_Description": injuries_created, "Treatment": []},
        [],
        {},
        role=current_user.role,
    )
    return {
        "patient": patient,
        "accident_record": accident,
        "injuries": patient_service.order_injuries(injuries_created),
        "severity": severity,
    }


@router.patch("/intake/{accident_id}")
def update_patient_intake(
    accident_id: UUID,
    form: UpdateIntakeForm,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_nurse),
):
    """
    Nurse progressively updates an intake form.
    Updates Patient, Accident_Record, and Injury_Description(s). No model is run.

    Basic details stay editable. Severity inputs are locked once the Severity
    decision is recorded, and inputs read only by the doctor models are locked
    once treatment is recorded; changing a locked field is refused with 409
    before anything is written.
    """
    # 1. Load the episode; one held by another hospital is "not found".
    accident = load_accident(supabase, current_user, accident_id, select=ACCIDENT_SELECT)
    accident_id = str(accident_id)
    patient_id = accident["patient_id"]

    severity = severity_workflow.load_status(supabase, accident, role=current_user.role)
    locked = severity_workflow.lock_violations(form, accident, severity)
    if locked:
        reasons = []
        if severity["locks"]["severity_inputs"]:
            reasons.append("Severity inputs are locked after the Severity decision")
        if severity["locks"]["later_model_inputs"]:
            reasons.append("inputs for the doctor's models are locked after treatment is recorded")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"These fields cannot be changed: {', '.join(locked)}. {'; '.join(reasons)}.",
        )
    for key in EMBEDDED_KEYS:
        accident.pop(key, None)
    try:
        patient, accident = _write_intake_update(supabase, form, accident, accident_id, patient_id)
    except APIError as exc:
        # The database re-checks the locks; a decision or treatment saved
        # since the check above (a concurrent request) makes them apply now.
        if exc.code == "23514":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                detail=f"These fields cannot be changed: {exc.message}.") from exc
        raise
    injuries_res = (
        supabase.table("Injury_Description")
        .select("*")
        .eq("accident_id", accident_id)
        .order("injury_order")
        .execute()
    )
    injuries = patient_service.order_injuries(injuries_res.data or [])

    # 5. No model runs on save. Report the Severity state for the saved values
    # (an earlier prediction is marked stale if its inputs have changed).
    saved = load_accident(supabase, current_user, accident_id, select=ACCIDENT_SELECT)
    severity = severity_workflow.load_status(supabase, saved, role=current_user.role)
    return {
        "patient": patient,
        "accident_record": accident,
        "injuries": injuries,
        "severity": severity,
    }


def _write_intake_update(supabase: Client, form: UpdateIntakeForm, accident: dict, accident_id: str, patient_id) -> tuple[dict, dict]:
    """Write the Patient, Accident_Record and injury changes of an intake update."""
    # 2. Update Patient
    patient_data = {}
    if form.full_name is not None: patient_data["full_name"] = form.full_name
    if form.age is not None: patient_data["age"] = form.age
    if form.gender is not None: patient_data["gender"] = form.gender
    
    patient = None
    if patient_data:
        patient_res = supabase.table("Patient").update(patient_data).eq("patient_id", patient_id).execute()
        if patient_res.data:
            patient = patient_res.data[0]
    
    if not patient:
        patient_res = supabase.table("Patient").select("*").eq("patient_id", patient_id).execute()
        patient = patient_res.data[0] if patient_res.data else {}

    # 3. Update Accident Record
    accident_data = {}
    if form.vehicle_type is not None: accident_data["vehicle_type"] = form.vehicle_type
    if form.helmet_worn is not None: accident_data["helmet_worn"] = form.helmet_worn
    if form.road_type is not None: accident_data["road_type"] = form.road_type
    if form.speed is not None: accident_data["speed"] = form.speed
    if form.incident_at is not None: accident_data["incident_at"] = form.incident_at.isoformat()
    if form.road_condition is not None: accident_data["road_condition"] = form.road_condition
    if form.visibility is not None: accident_data["visibility"] = form.visibility
    if form.road_signals_exist is not None: accident_data["road_signals_exist"] = form.road_signals_exist
    if form.collision_force_from is not None: accident_data["collision_force_from"] = form.collision_force_from
    if form.alcohol_consumption is not None: accident_data["alcohol_consumption"] = form.alcohol_consumption
    if form.illicit_drugs is not None: accident_data["illicit_drugs"] = form.illicit_drugs
    if form.mode_of_transport is not None: accident_data["mode_of_transport"] = form.mode_of_transport
    if form.time_to_hospital is not None: accident_data["time_to_hospital"] = form.time_to_hospital
    if form.collision_with is not None: accident_data["collision_with"] = form.collision_with
    if form.mode_of_travel_during_accident is not None: accident_data["mode_of_travel_during_accident"] = form.mode_of_travel_during_accident
    if form.passenger_type is not None: accident_data["passenger_type"] = form.passenger_type
    if form.other_injury is not None: accident_data["other_injury"] = form.other_injury

    if accident_data:
        accident_res = supabase.table("Accident_Record").update(accident_data).eq("accident_id", accident_id).execute()
        if accident_res.data:
            accident = accident_res.data[0]

    # 4. Update injuries in place, slot by slot. Each injury keeps its
    # injury_order (1 = primary, 2 = secondary) and its row id, so an edit can
    # never swap them. A slot missing from a non-empty list is removed; an
    # empty list leaves the injuries unchanged, as before.
    if form.injuries:
        supabase.table("Injury_Description").upsert(
            [
                {
                    "accident_id": accident_id,
                    "injury_order": injury.injury_order,
                    "site_of_injury": injury.site_of_injury,
                    "type_of_injury": injury.type_of_injury,
                }
                for injury in form.injuries
            ],
            on_conflict="accident_id,injury_order",
        ).execute()
        kept = sorted(injury.injury_order for injury in form.injuries)
        supabase.table("Injury_Description").delete().eq("accident_id", accident_id).not_.in_(
            "injury_order", kept
        ).execute()
    return patient, accident


@router.get("/my", response_model=list[dict])
def get_my_patients(
    response: Response,
    page: Page = Depends(page_params),
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_staff),
):
    """Return the latest accident records of the caller's hospital, whoever registered them.

    Each record carries ``severity_workflow``: the stored Severity prediction,
    the decision on it and the input/lock state. Nurses get only that; other
    staff also keep the display-only model outputs computed on read.
    """
    result = page.apply(
        supabase.table("Accident_Record")
        .select("*, Patient(*), Injury_Description(*), Treatment(*), Transfer(*)", count="exact")
        .eq("hospital_id", current_user.hospital_id)
        .order("created_at", desc=True)
    ).execute()
    Page.report_total(response, result.count)
    accidents = result.data or []
    workflows = severity_workflow.statuses_for(supabase, accidents, role=current_user.role)
    if current_user.role != NURSE:
        accidents = patient_service.add_severity_predictions(accidents)
    else:
        for accident in accidents:
            accident["Injury_Description"] = patient_service.order_injuries(accident.get("Injury_Description"))
    for accident in accidents:
        accident["severity_workflow"] = workflows[str(accident["accident_id"])]
    return [filter_predictions(current_user, accident) for accident in accidents]


# ── Transfer requests (operational; not model predictions or decisions) ──────

@router.get("/transfer-requests", response_model=list[TransferOut])
def get_transfer_requests(
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_nurse),
):
    """Transfer requests sent from the nurse's hospital: open ones first, then the latest closed.

    Every nurse of the sending hospital sees them, whoever registered the patient.
    """
    rows = (
        supabase.table("Transfer")
        .select("*")
        .eq("from_hospital_id", current_user.hospital_id)
        .order("requested_at", desc=True)
        .limit(100)
        .execute()
        .data
        or []
    )
    open_rows = [row for row in rows if row.get("status") in transfer_service.OPEN_STATUSES]
    closed_rows = [row for row in rows if row.get("status") not in transfer_service.OPEN_STATUSES][:20]
    return transfer_service.enrich(supabase, open_rows + closed_rows, current_user)


@router.post("/transfer-requests/{transfer_id}/process", response_model=TransferOut)
def process_transfer_request(
    transfer_id: UUID,
    body: ProcessTransferPayload,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_nurse),
):
    """Record whether an authorized transfer physically happened.

    "transferred" completes it: the destination becomes the episode's hospital
    and a new hospital stay starts. "not_transferred" closes it as rejected with
    the nurse's note. Either way the nurse is recorded as the processor.
    """
    if body.outcome == "transferred" and body.state_of_transfer not in transfer_service.TRANSFER_STATES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="state_of_transfer must be one of: " + "; ".join(transfer_service.TRANSFER_STATES),
        )
    transfer = transfer_service.enrich(supabase, [load_transfer(supabase, current_user, transfer_id)])[0]
    return transfer_service.transition(
        supabase,
        current_user,
        transfer,
        "complete" if body.outcome == "transferred" else "not_transferred",
        note=(body.note or "").strip() or None,
        state_of_transfer=body.state_of_transfer if body.outcome == "transferred" else None,
    )


@router.get("/transfer-states")
def get_transfer_states(current_user: Principal = Depends(require_nurse)):
    """How a patient can be moved, as recorded when completing a transfer."""
    return list(transfer_service.TRANSFER_STATES)
