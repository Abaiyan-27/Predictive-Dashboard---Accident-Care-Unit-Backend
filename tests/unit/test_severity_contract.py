"""The four severity labels are one contract shared by backend and frontend."""
import re
from pathlib import Path

import pytest

from app.contracts.severity import CONTRACT_PATH, SEVERITY_LABELS
from app.schemas.accident import SeverityPredictionOut

BACKEND_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = BACKEND_ROOT.parent / "Predictive-Dashboard---Accident-Care-Unit-Frontend"
FRONTEND_CONTRACT = FRONTEND_ROOT / "src" / "contracts" / "severity.ts"

EXPECTED = ("HIGH", "BORDERLINE", "LOW", "DEFER TO CLINICAL JUDGEMENT")
RETIRED = ("Priority", "Moderate", "Severity not classified")


def test_backend_contract_has_exactly_the_four_labels():
    assert SEVERITY_LABELS == EXPECTED
    assert CONTRACT_PATH.exists()


@pytest.mark.parametrize("label", EXPECTED)
def test_schema_accepts_contract_labels(label):
    assert SeverityPredictionOut(prediction=label, completeness=1, status="OK").prediction == label


@pytest.mark.parametrize("label", ["High", "Moderate", "Severity not classified", "Medium", None])
def test_schema_rejects_anything_else(label):
    with pytest.raises(ValueError):
        SeverityPredictionOut(prediction=label, completeness=1, status="OK")


@pytest.mark.skipif(not FRONTEND_CONTRACT.exists(), reason="frontend checkout not present")
def test_frontend_contract_matches_backend():
    source = FRONTEND_CONTRACT.read_text(encoding="utf-8")
    match = re.search(r"SEVERITY_LABELS\s*=\s*\[(.*?)\]\s*as const", source, re.S)
    assert match, "SEVERITY_LABELS array not found in severity.ts"
    labels = tuple(re.findall(r"'([^']+)'", match.group(1)))
    assert labels == SEVERITY_LABELS


def _source_files(root: Path, suffixes: tuple[str, ...]):
    return [
        path for path in root.rglob("*")
        if path.suffix in suffixes and "node_modules" not in path.parts
    ]


def test_backend_has_no_retired_severity_labels():
    # Code and contracts only: app/ml_models holds notebook outputs, including an
    # unused summary of an older three-class model.
    files = _source_files(BACKEND_ROOT / "app", (".py",)) + [CONTRACT_PATH]
    offenders = [
        f"{path.relative_to(BACKEND_ROOT)}: {word}"
        for path in files
        for word in RETIRED
        if word in path.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders


@pytest.mark.skipif(not FRONTEND_CONTRACT.exists(), reason="frontend checkout not present")
def test_frontend_has_no_retired_severity_labels():
    words = RETIRED + ("priority",)
    offenders = [
        f"{path.relative_to(FRONTEND_ROOT)}: {word}"
        for path in _source_files(FRONTEND_ROOT / "src", (".ts", ".tsx"))
        for word in words
        if word in path.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders
