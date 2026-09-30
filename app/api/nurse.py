"""Nurse router — patient intake and personal patient list."""
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from app.api.deps import get_current_user, require_nurse
from app.db.session import get_supabase
from app.schemas.accident import FullIntakeForm, UpdateIntakeForm
from app.services import audit_service, patient_service, schema_service

router = APIRouter(prefix="/patients", tags=["Nurse / Patients"])


@router.post("/intake", status_code=status.HTTP_201_CREATED)
def create_patient_intake(
    form: FullIntakeForm,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(require_nurse),
):
    """
    Nurse submits full intake form.
    Creates Patient, Accident_Record, Injury_Description(s), Hospital_Patient records,
    and runs ML severity prediction automatically.
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
    hospital_id = form.hospital_id or current_user.get("hospital_id")
    accident_data = {
        "patient_id": patient_id,
        "hospital_id": str(hospital_id) if hospital_id else None,
        "first_hospital_id": str(form.first_hospital_id) if form.first_hospital_id else (str(hospital_id) if hospital_id else None),
        "created_by": current_user.get("user_id"),
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
                "site_of_injury": injury.site_of_injury,
                "type_of_injury": injury.type_of_injury,
            }
            injury_result = supabase.table("Injury_Description").insert(injury_data).execute()
            if injury_result.data:
                injuries_created.append(injury_result.data[0])

    # 4. Create Hospital_Patient tracking record
    if hospital_id:
        supabase.table("Hospital_Patient").insert({
            "accident_id": accident_id,
            "hospital_id": str(hospital_id),
        }).execute()

    # 5. Audit log
    audit_service.log_action(
        supabase,
        actor_id=current_user.get("user_id"),
        role=current_user.get("role"),
        action_type="patient_intake",
        hospital_id=hospital_id,
        accident_id=accident_id,
    )

    # 6. Run every intake-time model whose artifact is installed. Missing
    # artifacts return UNAVAILABLE and never fall back to invented numbers.
    enriched_accident = patient_service.add_patient_predictions({
        **accident,
        "Patient": patient,
        "Injury_Description": injuries_created,
    })
    return {
        "patient": patient,
        "accident_record": accident,
        "injuries": injuries_created,
        "severity_prediction": enriched_accident["severity_prediction"],
        "stay_duration_prediction": enriched_accident["stay_duration_prediction"],
        "transfer_prediction": enriched_accident["transfer_prediction"],
    }


@router.patch("/intake/{accident_id}")
def update_patient_intake(
    accident_id: str,
    form: UpdateIntakeForm,
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(require_nurse),
):
    """
    Nurse progressively updates an intake form.
    Updates Patient, Accident_Record, and Injury_Description(s).
    """
    # 1. Get existing Accident Record to find patient_id
    accident_res = supabase.table("Accident_Record").select("*").eq("accident_id", accident_id).execute()
    if not accident_res.data:
        raise HTTPException(status_code=404, detail="Accident record not found.")
    accident = accident_res.data[0]
    patient_id = accident["patient_id"]

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

    # 4. Update Injuries (simplest way is delete and recreate if provided)
    injuries_created = []
    if form.injuries is not None and len(form.injuries) > 0:
        supabase.table("Injury_Description").delete().eq("accident_id", accident_id).execute()
        for injury in form.injuries:
            injury_data = {
                "accident_id": accident_id,
                "site_of_injury": injury.site_of_injury,
                "type_of_injury": injury.type_of_injury,
            }
            injury_result = supabase.table("Injury_Description").insert(injury_data).execute()
            if injury_result.data:
                injuries_created.append(injury_result.data[0])
    else:
        injuries_res = supabase.table("Injury_Description").select("*").eq("accident_id", accident_id).execute()
        injuries_created = injuries_res.data or []

    # 5. Run severity prediction automatically
    enriched_accident = patient_service.add_patient_predictions({
        **accident,
        "Patient": patient,
        "Injury_Description": injuries_created,
    })
    
    return {
        "patient": patient,
        "accident_record": accident,
        "injuries": injuries_created,
        "severity_prediction": enriched_accident["severity_prediction"],
        "stay_duration_prediction": enriched_accident["stay_duration_prediction"],
        "transfer_prediction": enriched_accident["transfer_prediction"],
    }


@router.get("/my", response_model=list[dict])
def get_my_patients(
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(get_current_user),
):
    """Return accident records created by the authenticated nurse, with patient info."""
    user_id = current_user.get("user_id") or current_user.get("officer_id")

    result = (
        supabase.table("Accident_Record")
        .select("*, Patient(*), Injury_Description(*), Treatment(*), Transfer(*)")
        .eq("created_by", str(user_id))
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )
    return patient_service.add_severity_predictions(result.data or [])
