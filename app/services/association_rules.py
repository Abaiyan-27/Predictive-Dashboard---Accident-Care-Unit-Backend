"""Association rules for the Government dashboard, from Notebooks/Association_Rule_Mining.ipynb.

The notebook mines the rules (Apriori, support 2%, confidence 50%, lift 1.2,
stability across 5 random splits, redundancy pruning) and, in its section 9,
rewrites each one in plain language. Its artifacts are installed unchanged in
app/ml_models/association_rules/:

- association_rules_readable.json: the plain-language rules (served here);
- association_rules_summary.json: the notebook's model_summary.json;
- association_rules.csv: the raw rules, kept for reference.

Nothing is mined at request time; the rules describe past accidents only and
contain no patient-level data.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from app.services import ml_service

FOLDER = "association_rules"
RULES_FILE = "association_rules_readable.json"
SUMMARY_FILE = "association_rules_summary.json"

# The notebook's strength bands (by lift), strongest first.
STRENGTHS = ("very strong", "strong", "moderate", "mild")
SORTS = ("lift", "confidence", "accidents")
DISCLAIMER = "These are associations found in past accidents, not proof of cause."
FIELDS = ("rank", "rule", "interpretation", "strength", "topic", "accidents",
          "confidence", "lift", "if_items", "then_items")


class RulesUnavailable(Exception):
    """The rules artifact is missing or invalid; the message is safe to show."""


_lock = threading.Lock()
_cache: dict[Path, tuple[float, dict]] = {}


def artifact_folder() -> Path:
    """The active model root's rules, else the ones packaged with the application."""
    for root in (ml_service.get_model_root(), ml_service.LOCAL_MODEL_ROOT):
        if (root / FOLDER / RULES_FILE).is_file():
            return root / FOLDER
    raise RulesUnavailable("Association rules are not installed. Run Association_Rule_Mining.ipynb first.")


def _items(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("empty item list")
    return [item.strip() for item in value.split(",") if item.strip()]


def _rule(record: dict) -> dict:
    rule = {
        "rank": int(record["rank"]),
        "rule": str(record["rule"]),
        "interpretation": str(record["interpretation"]),
        "strength": str(record["strength"]),
        "topic": str(record["topic"]),
        "accidents": int(record["accidents"]),
        "confidence": float(record["confidence"]),
        "lift": float(record["lift"]),
        "if_items": _items(record["if_items"]),
        "then_items": _items(record["then_items"]),
    }
    if rule["strength"] not in STRENGTHS or not 0 < rule["confidence"] <= 1 or rule["lift"] <= 0:
        raise ValueError("rule values out of range")
    return rule


def _read(folder: Path) -> dict:
    try:
        records = json.loads((folder / RULES_FILE).read_text(encoding="utf-8"))
        summary_path = folder / SUMMARY_FILE
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    except (OSError, json.JSONDecodeError) as exc:
        raise RulesUnavailable("The association rules could not be read.") from exc
    try:
        if not isinstance(records, list) or not records:
            raise ValueError("no rules")
        rules = [_rule(record) for record in records]
    except (KeyError, TypeError, ValueError) as exc:
        raise RulesUnavailable("The association rules artifact has an invalid format.") from exc
    rules.sort(key=lambda rule: rule["rank"])
    thresholds = summary.get("thresholds") if isinstance(summary.get("thresholds"), dict) else {}
    stability = summary.get("stability_check") if isinstance(summary.get("stability_check"), dict) else {}
    return {
        "rules": rules,
        "total_accidents": summary.get("total_rows"),
        "thresholds": {key: thresholds[key] for key in ("min_support", "min_confidence", "min_lift") if key in thresholds},
        "stability": {key: stability[key] for key in ("n_splits", "stability_floor") if key in stability},
    }


def load() -> dict:
    """The installed rules, re-read when the artifact changes."""
    folder = artifact_folder()
    try:
        modified = (folder / RULES_FILE).stat().st_mtime
    except OSError as exc:
        raise RulesUnavailable("The association rules could not be read.") from exc
    with _lock:
        cached = _cache.get(folder)
        if cached is None or cached[0] != modified:
            cached = (modified, _read(folder))
            _cache[folder] = cached
        return cached[1]


def query(*, topic: str = "", strength: str = "", search: str = "", sort_by: str = "lift", limit: int = 1000) -> dict:
    """Filter and sort the rules; ties keep the notebook's rank order."""
    data = load()
    rules = data["rules"]
    if topic:
        rules = [rule for rule in rules if rule["topic"].lower() == topic.lower()]
    if strength:
        rules = [rule for rule in rules if rule["strength"] == strength.lower()]
    if search:
        needle = search.lower()
        rules = [rule for rule in rules
                 if needle in rule["rule"].lower() or any(needle in item.lower() for item in rule["if_items"] + rule["then_items"])]
    sort_key = sort_by if sort_by in SORTS else "lift"
    rules = sorted(rules, key=lambda rule: (-rule[sort_key], rule["rank"]))

    topics: dict[str, int] = {}
    strengths = dict.fromkeys(STRENGTHS, 0)
    for rule in data["rules"]:
        topics[rule["topic"]] = topics.get(rule["topic"], 0) + 1
        strengths[rule["strength"]] += 1
    return {
        "status": "OK",
        "total_rules": len(data["rules"]),
        "matched_rules": len(rules),
        "total_accidents": data["total_accidents"],
        "thresholds": data["thresholds"],
        "stability": data["stability"],
        "topics": dict(sorted(topics.items(), key=lambda item: (-item[1], item[0]))),
        "strengths": strengths,
        "disclaimer": DISCLAIMER,
        "rules": rules[:limit],
        "message": None,
    }
