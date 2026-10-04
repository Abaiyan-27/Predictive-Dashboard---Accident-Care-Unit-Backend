"""Central role and hospital authorization.

Every patient-facing endpoint goes through this module so the rules live in
one place:

* Identity comes from the database row loaded for the token's subject
  (``User`` or ``Government_Officer``), never from claims inside the token.
* Staff (nurse, doctor, admin) act only within their own hospital. A staff
  account without a hospital is refused everywhere hospital data is involved.
* A patient/accident belongs to the hospital in ``Accident_Record.hospital_id``.
  Visibility is by hospital, not by which nurse created the record.
* Government officers have no hospital and no access to staff endpoints.
* Records from another hospital are reported as "not found", so IDs cannot be
  used to discover which records exist elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from fastapi import HTTPException, status
from supabase import Client

NURSE, DOCTOR, ADMIN, GOVERNMENT = "nurse", "doctor", "admin", "government"
STAFF_ROLES = frozenset({NURSE, DOCTOR, ADMIN})

ALL_MODELS = ("severity", "stay_duration", "transfer", "discharge")

# Model outputs each role may see. Nurses triage with severity only; doctors
# use all four; admins keep the read-only severity and bed-planning views the
# admin pages already show.
VISIBLE_MODELS: dict[str, frozenset[str]] = {
    NURSE: frozenset({"severity"}),
    DOCTOR: frozenset(ALL_MODELS),
    ADMIN: frozenset({"severity", "stay_duration", "transfer"}),
}

# Who may record a final decision on each model's prediction. Mirrors
# ML_Model.decision_roles in the database, which enforces the same rule.
DECISION_ROLES: dict[str, frozenset[str]] = {
    "severity": frozenset({NURSE}),
    "stay_duration": frozenset({DOCTOR}),
    "transfer": frozenset({DOCTOR}),
    "discharge": frozenset({DOCTOR}),
}

# Response keys that carry each model's output on accident/intake payloads.
PREDICTION_KEYS = {
    "severity": ("severity_prediction", "severity"),
    "stay_duration": ("stay_duration_prediction", "stay_duration"),
    "transfer": ("transfer_prediction", "transfer"),
    "discharge": ("discharge_prediction", "discharge"),
}

NOT_FOUND_DETAIL = {
    "patient": "Patient not found.",
    "accident": "Accident record not found.",
    "prediction": "Prediction not found.",
    "transfer": "Transfer request not found.",
}


@dataclass(frozen=True)
class Principal:
    """The authenticated caller, as currently recorded in the database."""

    id: str
    role: str
    hospital_id: Optional[str]
    is_government: bool

    @property
    def is_staff(self) -> bool:
        return not self.is_government and self.role in STAFF_ROLES


def principal_from_user(user: dict[str, Any]) -> Principal:
    """Build a Principal from the row get_current_user loaded."""
    if user.get("_type") == "government_officer":
        return Principal(
            id=str(user["officer_id"]), role=GOVERNMENT, hospital_id=None, is_government=True
        )
    hospital_id = user.get("hospital_id")
    return Principal(
        id=str(user["user_id"]),
        role=str(user.get("role") or ""),
        hospital_id=str(hospital_id) if hospital_id else None,
        is_government=False,
    )


def exact_ilike(value: str) -> str:
    """ILIKE pattern for a case-insensitive exact match (escapes LIKE wildcards; "_" is common in emails)."""
    return value.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# ── Errors ─────────────────────────────────────────────────────────────────

def forbidden(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


def not_found(kind: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NOT_FOUND_DETAIL[kind])


# ── Role and hospital checks ───────────────────────────────────────────────

def ensure_role(principal: Principal, roles: Iterable[str]) -> None:
    """Government access requires a Government_Officer record; staff roles require a User record."""
    allowed = frozenset(roles)
    permitted = (
        (principal.is_government and GOVERNMENT in allowed)
        or (principal.is_staff and principal.role in allowed)
    )
    if not permitted:
        raise forbidden(f"Access denied. Required roles: {sorted(allowed)}")


def ensure_assigned_hospital(principal: Principal) -> str:
    if not principal.is_staff:
        raise forbidden("Only hospital staff can access patient records.")
    if not principal.hospital_id:
        raise forbidden(
            "Your account is not assigned to a hospital. "
            "Ask a database administrator to assign one."
        )
    return principal.hospital_id


def belongs_to_hospital(principal: Principal, hospital_id: Any) -> bool:
    return bool(
        principal.is_staff
        and principal.hospital_id
        and hospital_id is not None
        and str(hospital_id) == principal.hospital_id
    )


def ensure_same_hospital(principal: Principal, hospital_id: Any, *, kind: str) -> None:
    """Refuse a record held by another hospital, as if it did not exist."""
    ensure_assigned_hospital(principal)
    if not belongs_to_hospital(principal, hospital_id):
        raise not_found(kind)


def ensure_own_hospital_id(principal: Principal, hospital_id: Any, field: str = "hospital_id") -> str:
    """A hospital id supplied by the client must be the caller's own hospital."""
    own = ensure_assigned_hospital(principal)
    if hospital_id is not None and str(hospital_id) != own:
        raise forbidden(f"{field} must be your own hospital.")
    return own


# ── Model visibility and decisions ─────────────────────────────────────────

def can_view_model(principal: Principal, model_key: str) -> bool:
    return principal.is_staff and model_key in VISIBLE_MODELS.get(principal.role, frozenset())


def ensure_can_view_model(principal: Principal, model_key: str) -> None:
    if not can_view_model(principal, model_key):
        raise forbidden(f"Your role cannot access the {model_key} model.")


def filter_predictions(principal: Principal, payload: dict[str, Any]) -> dict[str, Any]:
    """Drop model outputs the caller's role may not see from a response dict."""
    for model_key, keys in PREDICTION_KEYS.items():
        if not can_view_model(principal, model_key):
            for key in keys:
                payload.pop(key, None)
    return payload


def ensure_can_decide(principal: Principal, prediction: dict[str, Any], accident: dict[str, Any]) -> None:
    """Authorize recording a final decision on a stored prediction.

    The decider must hold the model's decision role, belong to the hospital
    where the prediction was made, and that hospital must still hold the
    episode. Admins and government officers can never decide.
    """
    model_key = str(prediction.get("model_key"))
    if principal.role not in DECISION_ROLES.get(model_key, frozenset()) or not principal.is_staff:
        raise forbidden(f"Your role cannot record decisions for the {model_key} model.")
    ensure_same_hospital(principal, accident.get("hospital_id"), kind="prediction")
    if not belongs_to_hospital(principal, prediction.get("hospital_id")):
        raise forbidden("Decisions can only be recorded by the hospital that made the prediction.")


# ── Scoped loaders ─────────────────────────────────────────────────────────

def load_accident(
    supabase: Client, principal: Principal, accident_id: Any, select: str = "*"
) -> dict[str, Any]:
    """Return an accident record held by the caller's hospital, else 404."""
    ensure_assigned_hospital(principal)
    rows = (
        supabase.table("Accident_Record")
        .select(select)
        .eq("accident_id", str(accident_id))
        .limit(1)
        .execute()
        .data
    )
    if not rows:
        raise not_found("accident")
    ensure_same_hospital(principal, rows[0].get("hospital_id"), kind="accident")
    return rows[0]


def load_patient(
    supabase: Client, principal: Principal, patient_id: Any, accident_select: str = "*"
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a patient and only the accident records held by the caller's hospital.

    A patient with no accident at the caller's hospital is "not found".
    """
    hospital_id = ensure_assigned_hospital(principal)
    accidents = (
        supabase.table("Accident_Record")
        .select(accident_select)
        .eq("patient_id", str(patient_id))
        .eq("hospital_id", hospital_id)
        .order("created_at", desc=True)
        .execute()
        .data
        or []
    )
    if not accidents:
        raise not_found("patient")
    patients = supabase.table("Patient").select("*").eq("patient_id", str(patient_id)).execute().data
    if not patients:
        raise not_found("patient")
    return patients[0], accidents


def load_prediction(supabase: Client, principal: Principal, prediction_id: Any) -> dict[str, Any]:
    """Return a stored prediction whose episode the caller's hospital holds and whose model it may see."""
    ensure_assigned_hospital(principal)
    rows = (
        supabase.table("Model_Prediction")
        .select("*")
        .eq("prediction_id", str(prediction_id))
        .limit(1)
        .execute()
        .data
    )
    if not rows:
        raise not_found("prediction")
    prediction = rows[0]
    load_accident(supabase, principal, prediction["accident_id"], select="accident_id,hospital_id")
    if not can_view_model(principal, prediction["model_key"]):
        raise not_found("prediction")
    return prediction


def transfer_involves_hospital(principal: Principal, transfer: dict[str, Any]) -> bool:
    return belongs_to_hospital(principal, transfer.get("from_hospital_id")) or belongs_to_hospital(
        principal, transfer.get("to_hospital_id")
    )


def load_transfer(supabase: Client, principal: Principal, transfer_id: Any) -> dict[str, Any]:
    """Return a transfer that starts or ends at the caller's hospital, else 404."""
    ensure_assigned_hospital(principal)
    rows = (
        supabase.table("Transfer").select("*").eq("transfer_id", str(transfer_id)).limit(1).execute().data
    )
    if not rows or not transfer_involves_hospital(principal, rows[0]):
        raise not_found("transfer")
    return rows[0]
