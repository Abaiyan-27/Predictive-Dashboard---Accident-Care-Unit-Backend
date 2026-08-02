"""Government router — accident trends and regional risk predictions."""
import random
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import require_government
from app.db.session import get_db
from app.models.user import User
from app.schemas.analytics import AccidentTrend, RiskPrediction
from app.services import ml_service

router = APIRouter(prefix="/government", tags=["Government"])

# Month labels cycling backwards from the current period
_MONTH_LABELS = [
    "Sep 2025", "Oct 2025", "Nov 2025", "Dec 2025",
    "Jan 2026", "Feb 2026", "Mar 2026", "Apr 2026",
    "May 2026", "Jun 2026", "Jul 2026", "Aug 2026",
]


@router.get("/accident-trends", response_model=list[AccidentTrend])
def get_accident_trends(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_government),
):
    """
    Return 12-month aggregated accident trend data.

    In production, aggregate from the patients table grouped by month.
    The heuristic below provides realistic-looking demo data.
    """
    # Simulate seasonal variation: spikes in winter + year-end
    base_counts = [145, 162, 138, 198, 173, 144, 130, 118, 128, 151, 169, 182]
    trends = [
        AccidentTrend(
            month=_MONTH_LABELS[i],
            count=base_counts[i] + random.randint(-8, 8),
        )
        for i in range(len(_MONTH_LABELS))
    ]
    return trends


@router.get("/risk-predictions", response_model=list[RiskPrediction])
def get_risk_predictions(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_government),
):
    """Return ML-predicted regional accident risk levels."""
    regions = ml_service.predict_regional_risk()
    return [
        RiskPrediction(
            region=r.region,
            risk_level=r.risk_level,
            predicted_accidents=r.predicted_accidents,
        )
        for r in regions
    ]
