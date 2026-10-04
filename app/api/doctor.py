"""Doctor router — hospital patients, patient detail, prognosis and prediction history.

All record access goes through app.core.authorization, so every endpoint is
limited to the caller's own hospital.
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from postgrest.exceptions import APIError
from supabase import Client

from app.api.deps import Page, page_params, require_doctor, require_nurse, require_staff
from app.core.authorization import (
    DOCTOR,
    Principal,
    can_view_model,
    ensure_own_hospital_id,
    filter_predictions,
    forbidden,
    load_accident,
    load_patient,
    load_prediction,
)
from app.db.session import get_supabase
from app.schemas.accident import TreatmentCreatePayload, TreatmentOut
from app.services import audit_service, doctor_workflow, ml_service, patient_service, prediction_store, severity_workflow

router = APIRouter(tags=["Doctor / Patients"])


ACCIDENT_DETAIL = "*, Patient(*), Injury_Description(*), Treatment(*), Transfer(*)"


@router.get("/patients/hospital/{hospital_id}")
def get_hospital_patients(
    hospital_id: UUID,
    response: Response,
    page: Page = Depends(page_params),
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_staff),
):
    """Return the latest accident records of the caller's own hospital.

    Doctors get the stored state only: ``severity_workflow`` (prediction and
    the nurse's decision) and ``doctor_workflow`` (each doctor model's state).
    Admins keep the display-only outputs computed on read.
    """
    own_hospital = ensure_own_hospital_id(current_user, hospital_id)
    result = page.apply(
        supabase.table("Accident_Record")
        .select(ACCIDENT_DETAIL, count="exact")
        .eq("hospital_id", own_hospital)
        .order("created_at", desc=True)
    ).execute()
    Page.report_total(response, result.count)
    if current_user.role == DOCTOR:
        accidents = result.data or []
        severity = severity_workflow.statuses_for(supabase, accidents, role=current_user.role)
        doctor = doctor_workflow.summaries_for(supabase, accidents, role=current_user.role)
        for accident in accidents:
            accident["Injury_Description"] = patient_service.order_injuries(accident.get("Injury_Description"))
            accident["severity_workflow"] = severity[str(accident["accident_id"])]
            accident["doctor_workflow"] = doctor[str(accident["accident_id"])]
        return accidents
    return [
        filter_predictions(current_user, accident)
        for accident in patient_service.add_severity_predictions(result.data or [])
    ]


@router.get("/patients/{patient_id}")
def get_patient(
    patient_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_staff),
):
    """Return a patient with only the accident records held by the caller's hospital."""
    patient, accidents = load_patient(supabase, current_user, patient_id, ACCIDENT_DETAIL)
    patient["accident_records"] = [
        filter_predictions(current_user, accident)
        for accident in patient_service.add_severity_predictions(accidents)
    ]

    socio_result = (
        supabase.table("Socioeconomic_Info")
        .select("*")
        .eq("patient_id", str(patient_id))
        .execute()
    )
    patient["socioeconomic_info"] = socio_result.data[0] if socio_result.data else None
    return patient


@router.get("/accidents/{accident_id}")
def get_accident_detail(
    accident_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_staff),
):
    """Return full accident record with patient, injuries, and treatments."""
    accident = load_accident(supabase, current_user, accident_id, ACCIDENT_DETAIL)
    return filter_predictions(current_user, patient_service.add_severity_prediction(accident))


@router.post("/treatments", status_code=status.HTTP_201_CREATED)
def create_treatment(
    payload: TreatmentCreatePayload,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_nurse),
):
    """A nurse records a treatment for an accident record of their own hospital.

    Treatment is an input of the Stay, Transfer and Discharge models, and every
    model input is entered by a nurse; doctors review the predictions and make
    the decisions. The hospital comes from the nurse's account. ``decided_by``
    may name the doctor who decided the treatment (a doctor of the same
    hospital). Treatment is refused (409) once any doctor-model decision
    exists, since it would change the inputs of a decided prediction. After
    saving, the undecided doctor models whose inputs are now complete run and
    their predictions are stored.
    """
    own_hospital = ensure_own_hospital_id(current_user, payload.hospital_id)
    episode = load_accident(supabase, current_user, payload.accident_id, select=doctor_workflow.ACCIDENT_SELECT)
    models = doctor_workflow.load_models_status(supabase, episode, role=current_user.role)
    if doctor_workflow.treatment_status(episode, models, role=current_user.role)["locked"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Treatment is locked: a doctor-model decision has already been recorded for this patient.",
        )

    decided_by = str(payload.decided_by) if payload.decided_by else None
    if decided_by is not None:
        decider = (
            supabase.table("User").select("role,hospital_id").eq("user_id", decided_by).execute().data
        )
        if not decider or decider[0].get("role") != DOCTOR or str(decider[0].get("hospital_id")) != own_hospital:
            raise forbidden("decided_by must be a doctor at your hospital.")

    treatment_data = {
        "accident_id": str(payload.accident_id),
        "hospital_id": own_hospital,
        "ward_number": payload.ward_number,
        "reason": payload.reason,
        "number_of_days_stay": payload.number_of_days_stay,
        "investigation_done": payload.investigation_done,
        "management_name_1": payload.management_name_1,
        "management_name_2": payload.management_name_2,
        "decided_by": decided_by,
        "logged_by": current_user.id,
    }
    try:
        result = supabase.table("Treatment").insert(treatment_data).execute()
    except APIError as exc:
        # The database refuses treatment once a doctor-model decision exists
        # (including one saved by a concurrent request after the check above).
        if exc.code == "23514":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=exc.message) from exc
        raise
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create treatment.")

    treatment = result.data[0]
    audit_service.log_event(
        supabase,
        actor_id=current_user.id,
        role=current_user.role,
        action_type="treatment_logged",
        hospital_id=own_hospital,
        accident_id=payload.accident_id,
        details={
            "treatment_id": treatment.get("treatment_id"),
            "number_of_days_stay": treatment.get("number_of_days_stay"),
            "investigation_done": treatment.get("investigation_done"),
            "management_name_1": treatment.get("management_name_1"),
            "management_name_2": treatment.get("management_name_2"),
            "decided_by": treatment.get("decided_by"),
        },
    )

    # Treatment completes the post-treatment inputs: run the doctor models
    # (never Severity, which is the nurse's) and store their predictions.
    accident = load_accident(supabase, current_user, payload.accident_id, select=doctor_workflow.ACCIDENT_SELECT)
    doctor_workflow.run_models(
        supabase, accident, actor_id=current_user.id, role=current_user.role, trigger_event="treatment_recorded"
    )
    return {
        **treatment,
        "doctor_workflow": doctor_workflow.load_models_status(supabase, accident, role=current_user.role),
    }


@router.get("/accidents/{accident_id}/treatment")
def get_treatment_status(
    accident_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_nurse),
):
    """The nurse's view of an episode's treatment: recorded entries, lock, and whether more can be added."""
    episode = load_accident(supabase, current_user, accident_id, select=doctor_workflow.ACCIDENT_SELECT)
    models = doctor_workflow.load_models_status(supabase, episode, role=current_user.role)
    return doctor_workflow.treatment_status(episode, models, role=current_user.role)


@router.get("/patients/{patient_id}/prognosis")
def get_patient_prognosis(
    patient_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_doctor),
):
    """Doctor-model outputs for the patient's latest episode at the doctor's hospital.

    Only models whose inputs are complete are run and stored (de-duplicated, so
    re-opening an unchanged patient adds no rows); the others report the
    model's own REFUSED answer without being stored. Severity is not run here:
    it is the nurse's model, so the latest stored Severity prediction is returned.
    """
    _, accidents = load_patient(supabase, current_user, patient_id, doctor_workflow.ACCIDENT_SELECT)
    accident = accidents[0]
    record = doctor_workflow.model_record(accident)
    predictions = patient_service.run_models(record, doctor_workflow.DOCTOR_MODELS)
    ready = {
        key: output for key, output in predictions.items()
        if ml_service.bundle_readiness(key, record)["ready"]
    }
    prediction_store.store_predictions(
        supabase,
        accident_id=accident["accident_id"],
        hospital_id=accident.get("hospital_id"),
        record=record,
        predictions=ready,
        trigger_event="prognosis_viewed",
        requested_by=current_user.id,
    )
    # Never return a scored result that is not in the prediction history: it
    # could be shown, but no decision could ever reference it.
    if any(not output.get("prediction_id") for output in ready.values()):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="A prediction could not be saved. Please try again.")
    severity = severity_workflow.load_status(supabase, accident, role=current_user.role)
    return {
        "severity": severity["prediction"],
        "stay_duration": predictions["stay_duration"],
        "transfer": predictions["transfer"],
        "discharge": predictions["discharge"],
    }


# ── Stored prediction history (read-only) ──────────────────────────────────

def _with_decisions(supabase: Client, predictions: list[dict]) -> list[dict]:
    if not predictions:
        return predictions
    decisions = (
        supabase.table("Prediction_Decision")
        .select("*")
        .in_("prediction_id", [row["prediction_id"] for row in predictions])
        .order("decision_seq")
        .execute()
        .data
        or []
    )
    for row in predictions:
        row["decisions"] = [d for d in decisions if d["prediction_id"] == row["prediction_id"]]
    return predictions


@router.get("/accidents/{accident_id}/predictions")
def get_prediction_history(
    accident_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_staff),
):
    """Every stored prediction for an episode, oldest first, limited to models the role may see."""
    load_accident(supabase, current_user, accident_id, select="accident_id,hospital_id")
    rows = (
        supabase.table("Model_Prediction")
        .select("*")
        .eq("accident_id", str(accident_id))
        .order("prediction_seq")
        .execute()
        .data
        or []
    )
    visible = [row for row in rows if can_view_model(current_user, row["model_key"])]
    return _with_decisions(supabase, visible)


@router.get("/predictions/{prediction_id}")
def get_prediction(
    prediction_id: UUID,
    supabase: Client = Depends(get_supabase),
    current_user: Principal = Depends(require_staff),
):
    """One stored prediction with its decisions."""
    prediction = load_prediction(supabase, current_user, prediction_id)
    return _with_decisions(supabase, [prediction])[0]
