"""Operational transfer requests: admin requests, nurse processes.

A transfer request is an operational event. It is not a model prediction and
not a doctor's decision: nothing here reads or writes Model_Prediction or
Prediction_Decision. When a transfer completes, the episode's transfer inputs
change (patient_service._transfer_inputs), so earlier Stay/Discharge
predictions become stale through the normal input-hash check and the doctor
runs the models again; recorded decisions stay as they were.

Statuses (the existing four, see migration 202610060001):
  pending_authorization -> authorized | rejected     (sending-hospital admin)
  authorized            -> approved | rejected       (sending-hospital nurse:
                                                      transferred / not transferred)
Every status change goes through the database function transfer_transition,
which locks the row, checks the actor again and uses the database clock.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from fastapi import HTTPException, status
from postgrest.exceptions import APIError
from supabase import Client

from app.core.authorization import Principal
from app.services import audit_service

PENDING, AUTHORIZED, APPROVED, REJECTED = "pending_authorization", "authorized", "approved", "rejected"
OPEN_STATUSES = (PENDING, AUTHORIZED)

# How the patient was moved: the "State of Transfer" values the transfer,
# stay and discharge notebooks were trained on (spelling kept as in the data).
TRANSFER_STATES: tuple[str, ...] = (
    "tranfer without any support",
    "spinal board used during transfer",
    "cervical collar used during transfer",
    "sinal board and cervical collar used during transfer",
    "sb,cc, oxygen support used during transfer",
    "oxygen used during transfer",
)

# action -> (role allowed, audit action, status it leads to)
ACTIONS = {
    "authorize": ("admin", "transfer_authorized", AUTHORIZED),
    "reject": ("admin", "transfer_rejected", REJECTED),
    "complete": ("nurse", "transfer_completed", APPROVED),
    "not_transferred": ("nurse", "transfer_not_completed", REJECTED),
}
ACTIONS_BY_STATUS = {
    PENDING: {"authorize", "reject"},
    AUTHORIZED: {"reject", "complete", "not_transferred"},
}
ACTOR_FIELDS = ("requested_by", "authorized_by", "rejected_by", "processed_by", "approved_by")

_HTTP_FROM_SQLSTATE = {"PT403": 403, "PT404": 404, "PT409": 409, "PT422": 422}


def allowed_actions(principal: Principal, transfer: dict[str, Any]) -> list[str]:
    """Actions this caller may take now: only staff of the sending hospital, by role."""
    if not principal.is_staff or str(transfer.get("from_hospital_id")) != principal.hospital_id:
        return []
    return sorted(
        action for action in ACTIONS_BY_STATUS.get(transfer.get("status"), set())
        if ACTIONS[action][0] == principal.role
    )


def network_hospitals(supabase: Client) -> list[dict[str, Any]]:
    return (
        supabase.table("Hospital")
        .select("hospital_id,name,district,type")
        .eq("is_network_hospital", True)
        .order("name")
        .execute()
        .data
        or []
    )


def _names(supabase: Client, table: str, key: str, ids: Iterable[Any]) -> dict[str, str]:
    wanted = sorted({str(value) for value in ids if value})
    if not wanted:
        return {}
    rows = supabase.table(table).select(f"{key},name").in_(key, wanted).execute().data or []
    return {str(row[key]): row.get("name") for row in rows}


def enrich(supabase: Client, transfers: list[dict[str, Any]], principal: Optional[Principal] = None) -> list[dict[str, Any]]:
    """Add patient, hospital and actor names (and the caller's allowed actions)."""
    if not transfers:
        return transfers
    hospitals = _names(
        supabase, "Hospital", "hospital_id",
        [t.get("from_hospital_id") for t in transfers] + [t.get("to_hospital_id") for t in transfers],
    )
    users = _names(supabase, "User", "user_id", [t.get(field) for t in transfers for field in ACTOR_FIELDS])
    accident_ids = sorted({str(t["accident_id"]) for t in transfers if t.get("accident_id")})
    patients: dict[str, str] = {}
    if accident_ids:
        for row in (
            supabase.table("Accident_Record").select("accident_id,Patient(full_name)")
            .in_("accident_id", accident_ids).execute().data or []
        ):
            patient = row.get("Patient")
            if isinstance(patient, list):
                patient = patient[0] if patient else None
            patients[str(row["accident_id"])] = (patient or {}).get("full_name")
    for transfer in transfers:
        transfer["patient_name"] = patients.get(str(transfer.get("accident_id")))
        transfer["source_hospital"] = hospitals.get(str(transfer.get("from_hospital_id")))
        transfer["destination_hospital"] = hospitals.get(str(transfer.get("to_hospital_id")))
        for field in ACTOR_FIELDS:
            transfer[f"{field}_name"] = users.get(str(transfer.get(field)))
        if principal is not None:
            transfer["allowed_actions"] = allowed_actions(principal, transfer)
    return transfers


def audit_details(transfer: dict[str, Any], *, old_status: Optional[str], new_status: str,
                  note: Optional[str] = None) -> dict[str, Any]:
    """Everything needed to answer "who requested it?" and "who processed it?"."""
    return {
        "transfer_id": str(transfer.get("transfer_id")),
        "accident_id": str(transfer.get("accident_id")),
        "patient_name": transfer.get("patient_name"),
        "from_hospital_id": str(transfer.get("from_hospital_id")),
        "from_hospital": transfer.get("source_hospital"),
        "to_hospital_id": str(transfer.get("to_hospital_id")),
        "to_hospital": transfer.get("destination_hospital"),
        "old_status": old_status,
        "new_status": new_status,
        "message": transfer.get("request_message"),
        "note": note,
        "requested_by": transfer.get("requested_by"),
        "requested_by_name": transfer.get("requested_by_name"),
        "authorized_by": transfer.get("authorized_by"),
        "authorized_by_name": transfer.get("authorized_by_name"),
        "rejected_by": transfer.get("rejected_by"),
        "rejected_by_name": transfer.get("rejected_by_name"),
        "processed_by": transfer.get("processed_by"),
        "processed_by_name": transfer.get("processed_by_name"),
        "state_of_transfer": transfer.get("state_of_transfer"),
    }


def transition(
    supabase: Client,
    principal: Principal,
    transfer: dict[str, Any],
    action: str,
    *,
    note: Optional[str] = None,
    state_of_transfer: Optional[str] = None,
) -> dict[str, Any]:
    """Apply one action through transfer_transition and audit it. Raises HTTP errors."""
    role, audit_action, _ = ACTIONS[action]
    if principal.role != role or str(transfer.get("from_hospital_id")) != principal.hospital_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"Only a {role} of the sending hospital can do this.")
    if action not in ACTIONS_BY_STATUS.get(transfer.get("status"), set()):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"Cannot {action.replace('_', ' ')} a transfer that is {transfer.get('status')}.")
    try:
        rows = supabase.rpc("transfer_transition", {
            "p_transfer_id": str(transfer["transfer_id"]),
            "p_action": action,
            "p_actor": principal.id,
            "p_note": note,
            "p_state_of_transfer": state_of_transfer,
        }).execute().data
    except APIError as exc:
        code = _HTTP_FROM_SQLSTATE.get(str(exc.code))
        if code:
            raise HTTPException(status_code=code, detail=exc.message) from exc
        raise
    updated = (rows if isinstance(rows, list) else [rows])[0] if rows else None
    if not updated:
        raise HTTPException(status_code=500, detail="The transfer was not updated.")
    updated = enrich(supabase, [updated], principal)[0]

    details = audit_details(updated, old_status=transfer.get("status"), new_status=updated["status"], note=note)
    audit_service.log_event(
        supabase, actor_id=principal.id, role=principal.role, action_type=audit_action,
        hospital_id=updated.get("from_hospital_id"), accident_id=updated.get("accident_id"), details=details,
    )
    if action == "complete":
        # The receiving hospital's audit log shows the arrival too.
        audit_service.log_event(
            supabase, actor_id=principal.id, role=principal.role, action_type="transfer_received",
            hospital_id=updated.get("to_hospital_id"), accident_id=updated.get("accident_id"), details=details,
        )
    return updated
