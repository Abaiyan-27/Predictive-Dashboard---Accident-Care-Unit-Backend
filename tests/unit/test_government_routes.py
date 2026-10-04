"""Static checks on the Government API surface and its response contract."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.routing import APIRoute
from pydantic import ValidationError

from app.api.deps import require_government
from app.main import app
from app.schemas.analytics import ModelSummary
from app.services.government_analytics import DECIDABLE_LABELS, MODEL_LABELS

STAFF_ONLY_PARAMS = {"hospital_id", "accident_id", "patient_id", "prediction_id", "transfer_id"}


def _dependencies(dependant):
    for dep in dependant.dependencies:
        yield dep.call
        yield from _dependencies(dep)


def _government_routes():
    return [r for r in app.routes if isinstance(r, APIRoute) and r.path.startswith("/api/government")]


def test_every_government_route_is_read_only():
    routes = _government_routes()
    # trends, historical trends, forecast, statistics, per-model summary, association rules
    assert len(routes) == 6
    for route in routes:
        assert route.methods == {"GET"}, route.path


def test_every_government_route_requires_a_government_officer():
    for route in _government_routes():
        calls = list(_dependencies(route.dependant))
        assert require_government in calls, route.path
        # No staff role checker anywhere in the dependency tree.
        staff_checkers = [c for c in calls if getattr(c, "allowed_roles", None) not in (None, ["government"])]
        assert staff_checkers == [], route.path


def test_government_routes_take_no_record_or_hospital_identifiers():
    for route in _government_routes():
        params = {p.name for p in route.dependant.path_params + route.dependant.query_params}
        assert not params & STAFF_ONLY_PARAMS, (route.path, params)


def test_no_staff_route_admits_government():
    for route in app.routes:
        if not isinstance(route, APIRoute) or route.path.startswith(("/api/government", "/api/auth")):
            continue
        for call in _dependencies(route.dependant):
            assert "government" not in (getattr(call, "allowed_roles", None) or []), route.path


def _summary(model_key="severity", **changes):
    labels = list(MODEL_LABELS[model_key])
    decidable = list(DECIDABLE_LABELS[model_key])
    data = {
        "model_key": model_key, "title": "Severity", "labels": labels, "decision_labels": decidable,
        "decision_role": "nurse", "total_predictions": 0, "episodes_with_prediction": 0,
        "current": [{"label": label, "count": 0} for label in labels], "unavailable": 0,
        "decisions": {"total": 0, "accepted": 0, "overridden": 0,
                      "distribution": [{"label": label, "count": 0} for label in decidable]},
        "monthly": [{"month": "Sep 2026", "total": 0, "counts": {label: 0 for label in labels}}],
        "by_hospital": [], "versions": [], "generated_at": datetime.now(timezone.utc),
    }
    data.update(changes)
    return data


def test_the_severity_contract_is_the_four_official_labels():
    assert MODEL_LABELS["severity"] == ("HIGH", "BORDERLINE", "LOW", "DEFER TO CLINICAL JUDGEMENT")
    assert DECIDABLE_LABELS["severity"] == ("HIGH", "BORDERLINE", "LOW")
    assert ModelSummary.model_validate(_summary())


@pytest.mark.parametrize("bad", ["Medium", "Moderate", "Priority", "Severity not classified", "High"])
def test_summaries_with_retired_or_unknown_labels_are_rejected(bad):
    labels = ["HIGH", "BORDERLINE", bad, "DEFER TO CLINICAL JUDGEMENT"]
    with pytest.raises(ValidationError):
        ModelSummary.model_validate(_summary(current=[{"label": label, "count": 0} for label in labels]))
    with pytest.raises(ValidationError):
        ModelSummary.model_validate(_summary(monthly=[{"month": "Sep 2026", "total": 0,
                                                       "counts": {label: 0 for label in labels}}]))


def test_severity_defer_is_never_a_decision_label():
    decidable = ["HIGH", "BORDERLINE", "LOW", "DEFER TO CLINICAL JUDGEMENT"]
    with pytest.raises(ValidationError):
        ModelSummary.model_validate(_summary(decision_labels=decidable))


@pytest.mark.parametrize("model_key,expected", [
    ("stay_duration", ("0-1 days", "2-3 days", "4+ days")),
    ("transfer", ("Transfer", "No transfer")),
    ("discharge", ("Complete recovery", "Partial recovery", "Further interventions")),
])
def test_doctor_model_contracts(model_key, expected):
    assert MODEL_LABELS[model_key] == expected == DECIDABLE_LABELS[model_key]
    assert ModelSummary.model_validate(_summary(model_key, decision_role="doctor"))
