"""The Government association rules are Association_Rule_Mining.ipynb's own output.

The installed plain-language file (the notebook's section 9) must describe
exactly the notebook's validated rules (association_rules.csv), carry its
strength bands and counts, and be served unchanged, filtered and sorted.
"""
from __future__ import annotations

import csv
import json
import shutil

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import government
from app.api.deps import require_government
from app.services import association_rules, ml_service

FOLDER = ml_service.LOCAL_MODEL_ROOT / association_rules.FOLDER


@pytest.fixture(autouse=True)
def packaged_models(monkeypatch):
    monkeypatch.setattr(ml_service, "MODEL_ROOT", ml_service.LOCAL_MODEL_ROOT)
    association_rules._cache.clear()


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(government.router, prefix="/api")
    app.dependency_overrides[require_government] = lambda: {"id": "officer", "role": "government"}
    return TestClient(app)


def readable():
    return json.loads((FOLDER / association_rules.RULES_FILE).read_text(encoding="utf-8"))


def mined():
    with (FOLDER / "association_rules.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def items(text):
    return frozenset(item.strip() for item in text.split(",") if item.strip())


def test_the_readable_rules_are_exactly_the_notebooks_validated_rules():
    summary = json.loads((FOLDER / association_rules.SUMMARY_FILE).read_text(encoding="utf-8"))
    plain, raw = readable(), mined()
    assert len(plain) == len(raw) == summary["final_non_redundant_rules"] == 897
    by_pair = {(items(r["antecedents"]), items(r["consequents"])): r for r in raw}
    assert len(by_pair) == len(raw)
    for rule in plain:
        source = by_pair[(items(rule["if_items"]), items(rule["then_items"]))]
        # Both files round the same values (to 3 and 2 places), so they agree within rounding.
        assert rule["confidence"] == pytest.approx(float(source["confidence"]), abs=1e-3)
        assert rule["lift"] == pytest.approx(float(source["lift"]), abs=6e-3)
        # accidents = support x all accidents (the CSV's support is rounded to 3 places)
        assert abs(rule["accidents"] - float(source["support"]) * summary["total_rows"]) <= 0.0005 * summary["total_rows"] + 1


def band(lift):
    return "very strong" if lift >= 3 else "strong" if lift >= 2 else "moderate" if lift >= 1.5 else "mild"


def test_strength_bands_and_ranks_follow_the_notebook():
    plain = readable()
    assert [rule["rank"] for rule in plain] == list(range(1, len(plain) + 1))
    assert all(a["lift"] >= b["lift"] for a, b in zip(plain, plain[1:]))
    for rule in plain:
        # Bands come from the unrounded lift; the stored lift has 2 places.
        assert rule["strength"] in {band(rule["lift"] - 0.005), band(rule["lift"] + 0.005)}, rule["rank"]
        # The sentences were written from the unrounded values: allow one unit of rounding.
        percent = int(rule["rule"].split(": in ", 1)[1].split("%", 1)[0])
        assert rule["rule"].startswith("When ") and abs(percent - rule["confidence"] * 100) <= 1
        times = float(rule["interpretation"].split("This is ", 1)[1].split(" times", 1)[0])
        assert abs(times - rule["lift"]) <= 0.1
        assert "=" not in rule["rule"]
    counts = {band: sum(rule["strength"] == band for rule in plain) for band in association_rules.STRENGTHS}
    assert counts == {"very strong": 5, "strong": 22, "moderate": 207, "mild": 663}   # the notebook's output


def test_no_rule_predicts_severity_from_itself():
    for rule in readable():
        assert not any(item.startswith("severity_label=") for item in rule["if_items"].split(", "))


def test_the_api_serves_the_rules_unchanged(client):
    response = client.get("/api/government/association-rules")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_rules"] == body["matched_rules"] == len(body["rules"]) == 897
    assert body["total_accidents"] == 6178
    assert body["thresholds"] == {"min_support": 0.02, "min_confidence": 0.5, "min_lift": 1.2}
    assert body["stability"] == {"n_splits": 5, "stability_floor": 0.6}
    assert body["strengths"] == {"very strong": 5, "strong": 22, "moderate": 207, "mild": 663}
    assert sum(body["topics"].values()) == 897
    assert "not proof of cause" in body["disclaimer"]
    first, source = body["rules"][0], readable()[0]
    assert first == {**source, "if_items": source["if_items"].split(", "), "then_items": source["then_items"].split(", ")}


def test_filters_and_sorting(client):
    def get(**params):
        response = client.get("/api/government/association-rules", params=params)
        assert response.status_code == 200, response.text
        return response.json()

    helmet = get(topic="helmet", strength="moderate")
    assert helmet["matched_rules"] == len(helmet["rules"]) > 0
    assert all(r["topic"] == "Helmet" and r["strength"] == "moderate" for r in helmet["rules"])
    pedestrians = get(search="Pedestrian")["rules"]
    assert pedestrians and all("pedestrian" in (r["rule"] + " ".join(r["if_items"] + r["then_items"])).lower() for r in pedestrians)
    by_accidents = get(sort_by="accidents", limit=50)["rules"]
    assert len(by_accidents) == 50 and all(a["accidents"] >= b["accidents"] for a, b in zip(by_accidents, by_accidents[1:]))
    assert get(sort_by="nonsense", limit=3)["rules"] == get(limit=3)["rules"]          # falls back to lift
    assert get(topic="no such topic")["rules"] == []
    assert client.get("/api/government/association-rules", params={"limit": 0}).status_code == 422


def test_missing_or_broken_artifacts_answer_503_without_paths(client, tmp_path, monkeypatch):
    monkeypatch.setattr(ml_service, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(ml_service, "LOCAL_MODEL_ROOT", tmp_path)
    missing = client.get("/api/government/association-rules")
    assert missing.status_code == 503 and "Association_Rule_Mining.ipynb" in missing.json()["detail"]

    folder = tmp_path / association_rules.FOLDER
    folder.mkdir()
    for content in ("not json", "[]", json.dumps([{"rank": 1, "rule": "x"}]),
                    json.dumps([{**readable()[0], "strength": "Moderate"}])):
        (folder / association_rules.RULES_FILE).write_text(content, encoding="utf-8")
        association_rules._cache.clear()
        response = client.get("/api/government/association-rules")
        assert response.status_code == 503, content
        assert str(tmp_path) not in response.text


def test_the_active_model_root_wins_and_changes_are_picked_up(tmp_path, monkeypatch):
    shutil.copytree(FOLDER, tmp_path / association_rules.FOLDER)
    monkeypatch.setattr(ml_service, "MODEL_ROOT", tmp_path)
    assert association_rules.artifact_folder() == tmp_path / association_rules.FOLDER
    path = tmp_path / association_rules.FOLDER / association_rules.RULES_FILE
    rules = json.loads(path.read_text(encoding="utf-8"))[:2]
    path.write_text(json.dumps(rules), encoding="utf-8")
    assert association_rules.query()["total_rules"] == 2
    # A model root without rules (e.g. a Hugging Face bundle) falls back to the packaged ones.
    shutil.rmtree(tmp_path / association_rules.FOLDER)
    assert association_rules.artifact_folder() == FOLDER
