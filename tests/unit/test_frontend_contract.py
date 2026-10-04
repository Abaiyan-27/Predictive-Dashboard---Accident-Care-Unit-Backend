"""The frontend's API wrappers and shared types still match this backend.

Reads the sibling frontend checkout (skipped when it is not present):
every path the frontend calls must be a backend route with that method, and
the role names, transfer statuses and Severity labels must be the backend's.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.routing import APIRoute

from app.contracts.severity import SEVERITY_LABELS
from app.core.authorization import ADMIN, DOCTOR, GOVERNMENT, NURSE
from app.main import app
from app.services import transfer_service

FRONTEND = Path(__file__).resolve().parents[3] / "Predictive-Dashboard---Accident-Care-Unit-Frontend" / "src"
pytestmark = pytest.mark.skipif(not FRONTEND.exists(), reason="frontend checkout not present")

RETIRED = ("Priority", "Moderate", "Medium severity", "Severity not classified")


def _normalise(path: str) -> str:
    path = path.split("?", 1)[0]
    # A trailing interpolation glued to a path segment is a query string (`${qs}`).
    path = re.sub(r"(?<=[^/])\$\{[^}]+\}$", "", path)
    path = re.sub(r"\$\{[^}]+\}", "{}", path)
    return re.sub(r"\{[^}]*\}", "{}", path)


def _frontend_calls() -> set[tuple[str, str]]:
    source = (FRONTEND / "api" / "index.ts").read_text(encoding="utf-8")
    calls = {
        (method.upper(), _normalise(path))
        for method, _, path in re.findall(
            r"apiClient\.(get|post|patch|put|delete)(?:<[^(]*?>)?\(\s*([`'])(.*?)\2", source, flags=re.S)
    }
    calls |= {("GET", _normalise(path)) for _, path in re.findall(r"getPage<[^(]*?>\(\s*([`'])(.*?)\1", source, flags=re.S)}
    return calls


def _backend_routes() -> set[tuple[str, str]]:
    routes = set()
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path.startswith("/api/"):
            for method in route.methods:
                routes.add((method, _normalise(route.path[len("/api"):])))
    return routes


def test_every_frontend_api_call_is_a_backend_route():
    calls = _frontend_calls()
    assert len(calls) >= 30, "the wrapper scan found too few calls; the regex needs updating"
    missing = sorted(calls - _backend_routes())
    assert missing == [], f"frontend calls endpoints the backend does not serve: {missing}"


def _union_members(name: str) -> list[str]:
    source = (FRONTEND / "types" / "index.ts").read_text(encoding="utf-8")
    match = re.search(rf"export type {name} = ([^;]+);", source)
    assert match, name
    return re.findall(r"'([^']+)'", match.group(1))


def test_role_names_match():
    assert sorted(_union_members("Role")) == sorted([NURSE, DOCTOR, ADMIN, GOVERNMENT])


def test_transfer_statuses_match():
    backend = [transfer_service.PENDING, transfer_service.AUTHORIZED, transfer_service.APPROVED, transfer_service.REJECTED]
    assert sorted(_union_members("TransferStatus")) == sorted(backend)


def test_severity_labels_match():
    source = (FRONTEND / "contracts" / "severity.ts").read_text(encoding="utf-8")
    block = re.search(r"SEVERITY_LABELS = \[(.*?)\]", source, flags=re.S).group(1)
    assert tuple(re.findall(r"'([^']+)'", block)) == SEVERITY_LABELS


def test_no_retired_terminology_in_the_frontend():
    offenders = []
    for path in FRONTEND.rglob("*.ts*"):
        text = path.read_text(encoding="utf-8")
        offenders += [f"{path.name}: {term}" for term in RETIRED if term in text]
    assert offenders == []


def test_frontend_no_longer_expects_unsaved_live_predictions():
    """Predictions shown in the UI come from stored history (workflow states), never computed-on-read fields."""
    source = (FRONTEND / "types" / "index.ts").read_text(encoding="utf-8") + (FRONTEND / "api" / "index.ts").read_text(encoding="utf-8")
    for field in ("severity_prediction", "stay_duration_prediction", "transfer_prediction", "predicted_los"):
        assert field not in source, field
