"""Persist model predictions as append-only history.

Each stored row records what one model returned for one accident: the output,
probabilities, status, the exact inputs it was given and the model version
that produced it. Rows are never updated; the database rejects UPDATE and
DELETE. Human decisions live in Prediction_Decision and reference these rows.

A new row is written only when something that determines the output changed
(model version, inputs, the output itself, or the hospital now holding the
episode), so re-opening a patient does not
grow the history with identical predictions.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

from supabase import Client

from app.services import ml_service

logger = logging.getLogger(__name__)

TRIGGER_EVENTS = {"intake_created", "intake_updated", "treatment_recorded", "prognosis_viewed"}

# (database URL, fingerprint) -> Model_Version.model_version_id
_version_ids: dict[tuple[str, str], str] = {}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _json_safe(value: Any) -> Any:
    """Turn NaN and numpy scalars into plain JSON values."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        return None if value != value else value  # NaN -> None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def model_inputs(model_key: str, record: dict[str, Any]) -> dict[str, Any]:
    """The subset of the model record that this model actually reads."""
    fields = ml_service.model_input_fields(model_key) or sorted(record)
    return {field: _json_safe(record.get(field)) for field in fields}


def inputs_hash(inputs: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(inputs).encode("utf-8")).hexdigest()


def ensure_model_version(supabase: Client, model_key: str) -> Optional[str]:
    """Return the Model_Version id for the artifact currently serving model_key."""
    identity = ml_service.model_fingerprint(model_key)
    if identity is None:
        return None
    fingerprint = identity["fingerprint"]
    cache_key = (str(getattr(supabase, "supabase_url", id(supabase))), fingerprint)
    if cache_key in _version_ids:
        return _version_ids[cache_key]

    def lookup() -> Optional[str]:
        rows = (
            supabase.table("Model_Version")
            .select("model_version_id")
            .eq("model_key", model_key)
            .eq("fingerprint", fingerprint)
            .limit(1)
            .execute()
            .data
        )
        return rows[0]["model_version_id"] if rows else None

    version_id = lookup()
    if version_id is None:
        supabase.table("Model_Version").upsert(
            identity, on_conflict="model_key,fingerprint", ignore_duplicates=True
        ).execute()
        version_id = lookup()
    if isinstance(version_id, str):
        _version_ids[cache_key] = version_id
    return version_id


def _same_output(previous: dict[str, Any], row: dict[str, Any]) -> bool:
    return all(
        previous.get(key) == row[key]
        for key in ("hospital_id", "model_version_id", "inputs_hash", "status", "prediction")
    )


def store_predictions(
    supabase: Client,
    *,
    accident_id: str,
    hospital_id: Optional[str],
    record: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
    trigger_event: str,
    requested_by: Optional[str],
) -> dict[str, Optional[str]]:
    """Store each model's result and return {model_key: prediction_id}.

    `predictions` maps model_key ("severity", "stay_duration", "transfer",
    "discharge") to the dict the API returns for that model. The returned ids
    are also written into those dicts as "prediction_id" so a later human
    decision can reference the exact prediction a clinician saw.
    """
    if trigger_event not in TRIGGER_EVENTS:
        raise ValueError(f"Unknown trigger_event: {trigger_event}")
    stored: dict[str, Optional[str]] = {key: None for key in predictions}
    if not accident_id or not hospital_id:
        logger.error("Prediction not stored: accident %s has no hospital", accident_id)
        return stored

    try:
        recent = (
            supabase.table("Model_Prediction")
            .select("prediction_id,model_key,hospital_id,model_version_id,inputs_hash,status,prediction")
            .eq("accident_id", str(accident_id))
            .order("prediction_seq", desc=True)
            .limit(4 * len(predictions) + 4)
            .execute()
            .data
            or []
        )
        latest: dict[str, dict[str, Any]] = {}
        for row in recent:
            latest.setdefault(row["model_key"], row)

        to_insert: list[dict[str, Any]] = []
        for model_key, result in predictions.items():
            version_id = ensure_model_version(supabase, model_key)
            if version_id is None:
                logger.error("Prediction not stored: no version identity for %s", model_key)
                continue
            inputs = model_inputs(model_key, record)
            row = {
                "accident_id": str(accident_id),
                "hospital_id": str(hospital_id),
                "model_key": model_key,
                "model_version_id": version_id,
                "status": result.get("status"),
                "prediction": result.get("prediction"),
                "probability": result.get("probability"),
                "probabilities": result.get("probabilities") or {},
                "confidence_level": result.get("confidence_level"),
                "completeness": result.get("completeness") or 0,
                "message": result.get("message"),
                "inputs": inputs,
                "inputs_hash": inputs_hash(inputs),
                "trigger_event": trigger_event,
                "requested_by": str(requested_by) if requested_by else None,
            }
            previous = latest.get(model_key)
            if previous and _same_output(previous, row):
                stored[model_key] = previous["prediction_id"]
            else:
                to_insert.append(row)

        if to_insert:
            inserted = supabase.table("Model_Prediction").insert(to_insert).execute().data or []
            for row in inserted:
                stored[row["model_key"]] = row["prediction_id"]
    except Exception:
        # Schema readiness is checked before intake writes; anything failing
        # here is logged loudly rather than discarding the clinical record.
        logger.exception("Failed to store model predictions for accident %s", accident_id)

    for model_key, prediction_id in stored.items():
        predictions[model_key]["prediction_id"] = prediction_id
    return stored
