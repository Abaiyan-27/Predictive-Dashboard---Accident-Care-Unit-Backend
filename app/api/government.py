"""Government router backed by real Supabase history and a forecast artifact."""
from collections import Counter
from datetime import datetime, timezone
import json

from fastapi import APIRouter, Depends, HTTPException, Query, status
from supabase import Client

from app.api.deps import require_government
from app.db.session import get_supabase
from app.schemas.analytics import AccidentTrend, AssociationRule, AssociationRulesResponse, ForecastResponse
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

@router.get("/association-rules", response_model=AssociationRulesResponse)
def get_association_rules(
    consequent: str = Query(default="", description="Filter rules whose consequents contain this substring"),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of rules to return"),
    sort_by: str = Query(default="lift", description="Sort column: lift, confidence, or support"),
    current_user: dict = Depends(require_government),
):
    """Return pre-computed Apriori association rules from the notebook artifact."""
    del current_user

    import csv
    from pathlib import Path as _Path

    artifact_path = ml_service.get_model_root() / "association_rules.csv"
    if not artifact_path.exists():
        # Fall back to the notebooks artifacts directory (project_root/Notebooks/artifacts/)
        _project_root = _Path(__file__).resolve().parent.parent.parent
        artifact_path = _project_root / "Notebooks" / "artifacts" / "association_rules.csv"

    if not artifact_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Association rules artifact is not installed. Run Association_Rule_Mining.ipynb first.",
        )

    try:
        with artifact_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except (OSError, Exception) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not read association rules artifact: {exc}",
        ) from exc

    rules: list[AssociationRule] = []
    for row in rows:
        try:
            antecedents = [item.strip() for item in row["antecedents"].split(",") if item.strip()]
            consequents = [item.strip() for item in row["consequents"].split(",") if item.strip()]
            # Apply consequent filter if provided
            if consequent and not any(consequent.lower() in c.lower() for c in consequents):
                continue
            rules.append(
                AssociationRule(
                    antecedents=antecedents,
                    consequents=consequents,
                    support=round(float(row["support"]), 4),
                    confidence=round(float(row["confidence"]), 4),
                    lift=round(float(row["lift"]), 4),
                )
            )
        except (KeyError, ValueError):
            continue

    # Sort
    valid_sorts = {"lift", "confidence", "support"}
    sort_key = sort_by if sort_by in valid_sorts else "lift"
    rules.sort(key=lambda r: getattr(r, sort_key), reverse=True)

    return AssociationRulesResponse(
        status="OK",
        total_rules=len(rules),
        thresholds={
            "min_support": 0.02,
            "min_confidence": 0.50,
            "min_lift": 1.2,
        },
        rules=rules[:limit],
        message=(
            f"{len(rules)} validated rules available (showing top {min(limit, len(rules))} by {sort_key}). "
            "Rules were checked for stability across 5 independent random data splits."
        ),
    )
