"""
ML Prediction Service
=====================
Implements a realistic heuristic/rules-based fallback engine.
All functions can be swapped with real model inference (sklearn, XGBoost,
PyTorch, etc.) by loading a .pkl / .joblib artifact and calling predict().

The function signatures and return types remain stable so callers never change.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

# ── Internal data classes ────────────────────────────────────────────────────


@dataclass
class SeverityPrediction:
    severity: str           # Critical | Moderate | Minor
    priority: str           # High | Medium | Low
    predicted_los_days: int


@dataclass
class PrognosisPrediction:
    expected_recovery_level: str        # Full Recovery | Partial Recovery | Long-term Care
    partial_recovery_probability: int   # 0–100
    transfer_recommendation: str


@dataclass
class RegionalRisk:
    region: str
    risk_level: str         # High | Medium | Low
    predicted_accidents: int


# ── Feature-extraction helpers ───────────────────────────────────────────────

_HIGH_SEVERITY_KEYWORDS = {
    "fracture", "hemorrhage", "bleeding", "trauma", "unconscious",
    "cardiac", "critical", "severe", "spine", "skull", "internal",
    "rupture", "crush", "penetrating",
}

_MODERATE_KEYWORDS = {
    "laceration", "concussion", "burn", "contusion", "dislocation",
    "whiplash", "sprain", "moderate", "broken",
}

_HIGH_RISK_COLLISION_TYPES = {
    "Multi-Vehicle Pileup", "Vehicle-Pedestrian", "Vehicle-Bicycle",
}

_MEDIUM_RISK_COLLISION_TYPES = {
    "Vehicle-Vehicle", "Single Vehicle",
}


def _score_injury(description: str, collision_type: str, age: int) -> int:
    """Return a raw risk score (0–100) used to bucket severity."""
    score = 0
    desc_lower = description.lower()

    # Keyword scoring
    for kw in _HIGH_SEVERITY_KEYWORDS:
        if kw in desc_lower:
            score += 20
    for kw in _MODERATE_KEYWORDS:
        if kw in desc_lower:
            score += 10

    # Collision type scoring
    if collision_type in _HIGH_RISK_COLLISION_TYPES:
        score += 25
    elif collision_type in _MEDIUM_RISK_COLLISION_TYPES:
        score += 10

    # Age-based vulnerability (young children & elderly get +10)
    if age < 12 or age > 65:
        score += 10

    return min(score, 100)


# ── Public API ───────────────────────────────────────────────────────────────


def predict_severity(
    age: int,
    gender: str,
    collision_type: str,
    injury_description: str,
) -> SeverityPrediction:
    """
    Predict patient severity, priority, and estimated length of stay.

    Replace the body with:
        model = joblib.load("models/severity_model.pkl")
        features = vectorise(age, gender, collision_type, injury_description)
        severity, priority, los = model.predict(features)
    when a trained model is available.
    """
    score = _score_injury(injury_description, collision_type, age)

    if score >= 45:
        severity = "Critical"
        priority = "High"
        los = random.randint(10, 20)
    elif score >= 20:
        severity = "Moderate"
        priority = "Medium"
        los = random.randint(3, 8)
    else:
        severity = "Minor"
        priority = "Low"
        los = random.randint(1, 3)

    return SeverityPrediction(
        severity=severity,
        priority=priority,
        predicted_los_days=los,
    )


def predict_prognosis(
    severity: str,
    age: int,
    collision_type: str,
    predicted_los_days: int,
) -> PrognosisPrediction:
    """
    Predict patient prognosis from stored patient attributes.

    Replace body with real inference model when available.
    """
    if severity == "Critical":
        if age > 60:
            level = "Long-term Care"
            prob = random.randint(20, 45)
            rec = (
                "Transfer to Regional Trauma Center recommended for specialised intensive care."
            )
        else:
            level = "Partial Recovery"
            prob = random.randint(45, 70)
            rec = (
                "Transfer to Regional Trauma Center recommended for orthopedic surgery."
                if collision_type in _HIGH_RISK_COLLISION_TYPES
                else "Monitor in current facility; transfer if complications arise."
            )
    elif severity == "Moderate":
        level = "Full Recovery"
        prob = random.randint(70, 88)
        rec = "No transfer needed; continue standard care protocol."
    else:  # Minor
        level = "Full Recovery"
        prob = random.randint(88, 98)
        rec = "No transfer needed; discharge expected within predicted LOS."

    return PrognosisPrediction(
        expected_recovery_level=level,
        partial_recovery_probability=prob,
        transfer_recommendation=rec,
    )


def predict_regional_risk() -> list[RegionalRisk]:
    """
    Return regional accident risk predictions.

    In production, replace with a trained geo-temporal model that ingests
    historical accident data, weather, traffic density, etc.
    """
    regions = [
        ("Downtown Core", "High", random.randint(70, 100)),
        ("Northern Highway Corridor", "High", random.randint(60, 90)),
        ("Eastern Industrial Zone", "Medium", random.randint(35, 60)),
        ("Western Residential", "Medium", random.randint(25, 50)),
        ("Southern Coastal Road", "Medium", random.randint(30, 55)),
        ("Central Business District", "Low", random.randint(10, 30)),
        ("Suburban South", "Low", random.randint(8, 25)),
        ("Airport Expressway", "High", random.randint(55, 85)),
    ]
    return [
        RegionalRisk(region=r, risk_level=rl, predicted_accidents=pa)
        for r, rl, pa in regions
    ]
