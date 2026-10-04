"""Network-wide, aggregate-only statistics for each patient model (Government).

Reads the persisted prediction and decision history; never runs a model and
never returns patient-level data (no ids, names or model inputs). Counts cover
every hospital: Government officers have no hospital and see the whole network.

Definitions used throughout:

* An episode's *current* prediction is its latest ``Model_Prediction`` row for
  the model (highest ``prediction_seq``). Earlier reruns stay in history but are
  not counted twice.
* An episode's *final decision* is its latest ``Prediction_Decision`` for the
  model (highest ``decision_seq``), on whichever of its predictions it was made.
* Monthly figures group episodes by admission month (``incident_at``, falling
  back to ``created_at``, as the accident-trend endpoint does).
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from supabase import Client

from app.contracts.severity import SEVERITY_DEFER, SEVERITY_LABELS
from app.services.doctor_workflow import DECISION_LABELS

PAGE_SIZE = 1000
TREND_MONTHS = 12

# Every model's valid outputs, in display order, from the existing contracts.
MODEL_LABELS: dict[str, tuple[str, ...]] = {"severity": SEVERITY_LABELS, **DECISION_LABELS}
# Outputs a person may record as a final decision (Severity's DEFER is the
# model's "cannot score" answer, never a decision).
DECIDABLE_LABELS: dict[str, tuple[str, ...]] = {
    "severity": tuple(label for label in SEVERITY_LABELS if label != SEVERITY_DEFER),
    **DECISION_LABELS,
}
MODEL_TITLES = {
    "severity": "Severity",
    "stay_duration": "Stay Duration",
    "transfer": "Transfer",
    "discharge": "Discharge Outcome",
}
DECISION_ROLE = {"severity": "nurse", "stay_duration": "doctor", "transfer": "doctor", "discharge": "doctor"}
TRANSFER_STATUSES = ("pending_authorization", "authorized", "approved", "rejected")


def _fetch_all(supabase: Client, table: str, columns: str, *, order: str, eq: Optional[tuple[str, str]] = None) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        query = supabase.table(table).select(columns)
        if eq:
            query = query.eq(*eq)
        page = query.order(order).range(offset, offset + PAGE_SIZE - 1).execute().data or []
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            return rows
        offset += PAGE_SIZE


def _parse(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _recent_months(now: datetime, count: int = TREND_MONTHS) -> list[tuple[int, int]]:
    year, month = now.year, now.month
    months: list[tuple[int, int]] = []
    for _ in range(count):
        months.append((year, month))
        month -= 1
        if month == 0:
            year, month = year - 1, 12
    return list(reversed(months))


def _distribution(counts: Counter, labels: Iterable[str]) -> list[dict[str, Any]]:
    labels = tuple(labels)
    total = sum(counts[label] for label in labels)
    return [
        {
            "label": label,
            "count": counts[label],
            "percentage": round(100 * counts[label] / total, 1) if total else None,
        }
        for label in labels
    ]


def _admission_months(supabase: Client) -> dict[str, tuple[int, int]]:
    try:
        rows = _fetch_all(supabase, "Accident_Record", "accident_id,incident_at,created_at", order="accident_id")
    except Exception:
        # Installations without the optional incident_at column.
        rows = _fetch_all(supabase, "Accident_Record", "accident_id,created_at", order="accident_id")
    months: dict[str, tuple[int, int]] = {}
    for row in rows:
        at = _parse(row.get("incident_at")) or _parse(row.get("created_at"))
        if at:
            months[str(row["accident_id"])] = (at.year, at.month)
    return months


def model_summary(supabase: Client, model_key: str, *, now: Optional[datetime] = None) -> dict[str, Any]:
    """Aggregate prediction and decision statistics for one model, network-wide."""
    labels = MODEL_LABELS[model_key]
    now = now or datetime.now(timezone.utc)

    predictions = _fetch_all(
        supabase, "Model_Prediction",
        "prediction_id,prediction_seq,accident_id,hospital_id,model_version_id,status,prediction",
        order="prediction_seq", eq=("model_key", model_key),
    )
    decisions = _fetch_all(
        supabase, "Prediction_Decision", "prediction_id,decision_seq,final_decision,decision_source",
        order="decision_seq", eq=("model_key", model_key),
    )

    # Rows arrive in sequence order, so the last one seen per episode is the latest.
    current: dict[str, dict] = {}
    episode_of: dict[str, str] = {}
    for row in predictions:
        accident = str(row["accident_id"])
        current[accident] = row
        episode_of[str(row["prediction_id"])] = accident
    final: dict[str, dict] = {}
    for row in decisions:
        accident = episode_of.get(str(row["prediction_id"]))
        if accident:
            final[accident] = row

    current_counts: Counter = Counter()
    unavailable = 0
    for row in current.values():
        if row.get("prediction") in labels:
            current_counts[row["prediction"]] += 1
        else:
            unavailable += 1   # the model could not score; not an output label

    # Monthly: current output per episode, by admission month.
    months = _recent_months(now)
    admitted = _admission_months(supabase) if current else {}
    monthly_counts: dict[tuple[int, int], Counter] = {month: Counter() for month in months}
    for accident, row in current.items():
        month = admitted.get(accident)
        if month in monthly_counts and row.get("prediction") in labels:
            monthly_counts[month][row["prediction"]] += 1

    # Per hospital: the hospital responsible when the current prediction was made.
    hospital_names = {
        str(h["hospital_id"]): h.get("name") or "Unknown hospital"
        for h in (supabase.table("Hospital").select("hospital_id,name").execute().data or [])
    }
    by_hospital: dict[str, Counter] = {}
    for row in current.values():
        if row.get("prediction") in labels:
            by_hospital.setdefault(str(row["hospital_id"]), Counter())[row["prediction"]] += 1

    # Final decisions and how often the model's output was kept.
    decision_counts: Counter = Counter(row["final_decision"] for row in final.values())
    accepted = sum(1 for row in final.values() if row.get("decision_source") == "MODEL")
    overridden = len(final) - accepted

    version_rows = (
        supabase.table("Model_Version")
        .select("model_version_id,contract_version,artifact_name,first_seen_at")
        .eq("model_key", model_key)
        .execute()
        .data
        or []
    )
    version_usage = Counter(str(row["model_version_id"]) for row in predictions)

    return {
        "model_key": model_key,
        "title": MODEL_TITLES[model_key],
        "labels": list(labels),
        "decision_labels": list(DECIDABLE_LABELS[model_key]),
        "decision_role": DECISION_ROLE[model_key],
        "total_predictions": len(predictions),
        "episodes_with_prediction": len(current),
        "current": _distribution(current_counts, labels),
        "unavailable": unavailable,
        "decisions": {
            "total": len(final),
            "accepted": accepted,
            "overridden": overridden,
            "acceptance_rate": round(100 * accepted / len(final), 1) if final else None,
            "distribution": _distribution(decision_counts, DECIDABLE_LABELS[model_key]),
        },
        "monthly": [
            {
                "month": datetime(year, month, 1).strftime("%b %Y"),
                "total": sum(monthly_counts[(year, month)].values()),
                "counts": {label: monthly_counts[(year, month)][label] for label in labels},
            }
            for year, month in months
        ],
        "by_hospital": sorted(
            (
                {
                    "hospital": hospital_names.get(hospital_id, "Unknown hospital"),
                    "total": sum(counts.values()),
                    "counts": {label: counts[label] for label in labels},
                }
                for hospital_id, counts in by_hospital.items()
            ),
            key=lambda item: (-item["total"], item["hospital"]),
        ),
        "versions": sorted(
            (
                {
                    "contract_version": row.get("contract_version"),
                    "artifact_name": row.get("artifact_name"),
                    "first_seen_at": row.get("first_seen_at"),
                    "predictions": version_usage[str(row["model_version_id"])],
                }
                for row in version_rows
            ),
            key=lambda item: item["first_seen_at"] or "",
            reverse=True,
        ),
        "generated_at": now,
    }


def transfers_by_status(supabase: Client) -> dict[str, int]:
    """Network-wide count of operational transfer requests in each status."""
    counts: dict[str, int] = {}
    for status_value in TRANSFER_STATUSES:
        result = (
            supabase.table("Transfer")
            .select("transfer_id", count="exact")
            .eq("status", status_value)
            .execute()
        )
        counts[status_value] = result.count or 0
    return counts
