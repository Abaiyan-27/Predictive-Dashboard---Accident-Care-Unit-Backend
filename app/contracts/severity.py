"""Single source of truth for the severity-triage output labels.

The labels come from ``severity_labels.json`` beside this module. Unpacking
into exactly four names makes the import fail if the file ever gains or loses
a label, so no other module can quietly work with a different set.
"""
from __future__ import annotations

import json
from pathlib import Path

CONTRACT_PATH = Path(__file__).with_name("severity_labels.json")

SEVERITY_LABELS: tuple[str, ...] = tuple(
    json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))["labels"]
)

SEVERITY_HIGH, SEVERITY_BORDERLINE, SEVERITY_LOW, SEVERITY_DEFER = SEVERITY_LABELS


def is_severity_label(value: object) -> bool:
    return isinstance(value, str) and value in SEVERITY_LABELS
