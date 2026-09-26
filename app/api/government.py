"""Government router backed by real Supabase history and a forecast artifact."""
from collections import Counter
from datetime import datetime, timezone
import json

from fastapi import APIRouter, Depends, HTTPException, Query, status
from supabase import Client

from app.api.deps import require_government
from app.db.session import get_supabase
from app.schemas.analytics import AccidentTrend, ForecastResponse
from app.services import ml_service

router = APIRouter(prefix="/government", tags=["Government"])


def _recent_months(count: int) -> list[tuple[int, int]]:
    now = datetime.now(timezone.utc)
    year, month = now.year, now.month
    result: list[tuple[int, int]] = []
    for _ in range(count):
        result.append((year, month))
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    return list(reversed(result))


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@router.get("/accident-trends", response_model=list[AccidentTrend])
def get_accident_trends(
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(require_government),
):
    """Aggregate the latest 12 calendar months from actual accident records."""
    del current_user
    months = _recent_months(12)
    counts: Counter[tuple[int, int]] = Counter()

    offset = 0
    page_size = 1000
    include_incident_at = True
    while True:
        try:
            result = (
                supabase.table("Accident_Record")
                .select("incident_at,created_at" if include_incident_at else "created_at")
                .range(offset, offset + page_size - 1)
                .execute()
            )
        except Exception:
            if not include_incident_at:
                raise
            # Existing installations can still show real created_at trends
            # before the optional incident_at migration is applied.
            include_incident_at = False
            offset = 0
            counts.clear()
            continue
        rows = result.data or []
        for row in rows:
            occurred_at = _parse_timestamp(row.get("incident_at")) or _parse_timestamp(row.get("created_at"))
            if occurred_at:
                counts[(occurred_at.year, occurred_at.month)] += 1
        if len(rows) < page_size:
            break
        offset += page_size

    return [
        AccidentTrend(
            month=datetime(year, month, 1).strftime("%b %Y"),
            count=counts[(year, month)],
        )
        for year, month in months
    ]


@router.get("/accident-historical-trends", response_model=list[AccidentTrend])
def get_accident_historical_trends(
    current_user: dict = Depends(require_government),
):
    """Return monthly history exported from the forecasting notebook dataset."""
    del current_user
    artifact_path = ml_service.get_model_root() / "forecasting" / "accident_historical_trends.json"
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Historical accident-trends artifact is not installed.",
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Historical accident-trends artifact could not be loaded.",
        ) from exc

    if not isinstance(payload, list):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Historical accident-trends artifact has an invalid format.",
        )
    return payload


@router.get("/accident-forecast", response_model=ForecastResponse)
def get_accident_forecast(
    horizon: int = Query(default=6, ge=1, le=24),
    current_user: dict = Depends(require_government),
):
    """Return future monthly counts from the installed forecasting artifact."""
    del current_user
    return ml_service.predict_accident_forecast(horizon).to_dict()


@router.get("/statistics")
def get_statistics(
    supabase: Client = Depends(get_supabase),
    current_user: dict = Depends(require_government),
):
    """Return aggregate statistics from the database."""
    del current_user
    accidents = supabase.table("Accident_Record").select("accident_id", count="exact").execute()
    patients = supabase.table("Patient").select("patient_id", count="exact").execute()
    hospitals = supabase.table("Hospital").select("hospital_id", count="exact").execute()
    transfers = supabase.table("Transfer").select("transfer_id", count="exact").execute()
    pending = (
        supabase.table("Transfer")
        .select("transfer_id", count="exact")
        .eq("status", "pending_authorization")
        .execute()
    )
    return {
        "total_accidents": accidents.count or 0,
        "total_patients": patients.count or 0,
        "total_hospitals": hospitals.count or 0,
        "total_transfers": transfers.count or 0,
        "pending_transfers": pending.count or 0,
    }
