"""Model loading and inference for every prediction shown by the dashboard.

Each deployed notebook artifact has a small ``model_contract.json`` beside it.
Missing artifacts, incomplete inputs, and inference failures are reported
explicitly; the service never substitutes demo or randomly generated values.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

from app.contracts.severity import (
    SEVERITY_BORDERLINE,
    SEVERITY_DEFER,
    SEVERITY_HIGH,
    SEVERITY_LOW,
)


LOCAL_MODEL_ROOT = Path(__file__).resolve().parent.parent / "ml_models"
MODEL_ROOT = LOCAL_MODEL_ROOT
MODEL_SOURCE = "local application bundle"

AGE_ORDER = [
    "1-9", "10-19", "20-29", "30-39", "40-49",
    "50-59", "60-69", "70-79", "80-89", "90-100",
]
TIME_ORDER = [
    "less than 15 minutes",
    "15 minutes - 30 minutes",
    "30 minutes - 1 hour",
    "1 hour - 2 hour",
    "more than 2 hour",
]
AGE_MAPPING = {value: index for index, value in enumerate(AGE_ORDER)}
TIME_MAPPING = {value: index for index, value in enumerate(TIME_ORDER)}

COMPLETENESS_FLOOR = 0.40
LOW_CONF_PROB = 0.60

# Fields documented only after the doctor records treatment. The stay, transfer
# and discharge notebooks were trained on records where these were filled in;
# scoring at intake would let the imputer stand in for the inputs that carry
# most of the signal, so these bundles refuse until they are recorded.
POST_TREATMENT_INPUTS = ("Investigation Done", "Management Name 1")
REQUIRED_INPUTS: dict[str, tuple[str, ...]] = {
    "stay_duration": POST_TREATMENT_INPUTS,
    "transfer": POST_TREATMENT_INPUTS,
    "discharge": POST_TREATMENT_INPUTS + ("total_days",),
}


@dataclass
class ModelPrediction:
    prediction: Optional[str]
    probability: Optional[float]
    probabilities: dict[str, float]
    completeness: float
    status: str
    confidence_level: Optional[str]
    model_name: str
    model_version: Optional[str]
    message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ForecastPoint:
    period: str
    predicted_accidents: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    risk_level: Optional[str] = None


@dataclass
class ForecastPrediction:
    status: str
    model_name: str
    model_version: Optional[str]
    points: list[ForecastPoint]
    message: Optional[str] = None
    risk_thresholds: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["points"] = [asdict(point) for point in self.points]
        return data


def _forecast_risk_thresholds(reference_values: list[float]) -> dict[str, Any]:
    """Create relative Low/Medium/High bands from historical monthly tertiles."""
    values = np.asarray(reference_values, dtype=float)
    medium_min = int(np.ceil(np.quantile(values, 1 / 3)))
    high_min = max(medium_min + 1, int(np.ceil(np.quantile(values, 2 / 3))))
    return {
        "medium_min": medium_min,
        "high_min": high_min,
        "method": "historical_monthly_tertiles",
    }


def _forecast_risk_level(value: float, thresholds: dict[str, Any]) -> str:
    if value >= float(thresholds["high_min"]):
        return "High"
    if value >= float(thresholds["medium_min"]):
        return "Medium"
    return "Low"


def _missing(value: Any) -> bool:
    if value is None or value == "":
        return True
    return isinstance(value, float) and np.isnan(value)


def _normalise_value(value: Any) -> Any:
    """Apply the notebooks' text cleaning: strip, lowercase, blank/'nan'/'none' -> missing."""
    if isinstance(value, str):
        cleaned = value.strip().lower()
        return None if cleaned in {"", "nan", "none"} else cleaned
    return value


def normalise_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _normalise_value(value) for key, value in record.items()}


def missing_required_inputs(record: dict[str, Any], settings: dict[str, Any]) -> list[str]:
    return [
        feature for feature in settings.get("required_inputs") or []
        if _missing(record.get(feature))
    ]


def _unavailable(model_name: str, version: Optional[str], message: str) -> ModelPrediction:
    return ModelPrediction(
        prediction=None,
        probability=None,
        probabilities={},
        completeness=0,
        status="UNAVAILABLE",
        confidence_level=None,
        model_name=model_name,
        model_version=version,
        message=message,
    )


# ---------------------------------------------------------------------------
# Severity-triage notebook bundle
# ---------------------------------------------------------------------------

def _severity_defer(
    model_name: str,
    version: Optional[str],
    status: str,
    message: str,
    completeness: float = 0,
) -> ModelPrediction:
    """Severity always answers with a contract label; when it cannot score, it defers."""
    return ModelPrediction(
        prediction=SEVERITY_DEFER,
        probability=None,
        probabilities={},
        completeness=round(completeness, 3),
        status=status,
        confidence_level=None,
        model_name=model_name,
        model_version=version,
        message=message,
    )


_HIGH_RISK_REGIONS = {"head_face", "spine", "chest", "abdomen", "pelvis"}
_NIGHT_WINDOWS = {"21:00 - 00:00", "00:00 - 03:00", "03:00 - 06:00"}
_VULNERABLE_USERS = {
    "motor cyclist",
    "motorbike pillion rider",
    "bicycle",
    "bicycle pillion",
    "pedestrian",
}
_HEAVY_COUNTERPARTS = {
    "heavy vehicle",
    "car or van",
    "three wheeler",
    "bus",
    "train",
}


def _severity_region(value: Any) -> Any:
    if _missing(value):
        return np.nan
    site = str(value).lower()
    if "no injury" in site or "no secondary" in site:
        return "none"
    if any(key in site for key in ("head", "forehead", "facial", "eye")):
        return "head_face"
    if any(key in site for key in ("cervical", "thoracic", "lumbar", "sacrum")):
        return "spine"
    if "chest" in site:
        return "chest"
    if "abdomen" in site:
        return "abdomen"
    if "pelvis" in site:
        return "pelvis"
    if any(
        key in site
        for key in ("shoulder", "clavicle", "humerus", "elbow", "forearm", "hand", "carpal")
    ):
        return "upper_limb"
    if any(
        key in site
        for key in ("thigh", "femur", "knee", "leg", "foot", "tarsal", "patella", "fibula", "tibia")
    ):
        return "lower_limb"
    return "other"


def _optional_membership(value: Any, choices: set[str]) -> float:
    if _missing(value):
        return np.nan
    return float(str(value) in choices)


def _engineer_severity_row(raw: pd.DataFrame) -> pd.DataFrame:
    row = raw.copy()
    row["injury_region_1"] = row["Site of Injury No1"].map(_severity_region)
    row["injury_region_2"] = row["Site of injury No 2"].map(_severity_region)
    both_regions_missing = row["injury_region_1"].isna() & row["injury_region_2"].isna()
    row["torso_head_injury"] = np.where(
        both_regions_missing,
        np.nan,
        (
            row["injury_region_1"].isin(_HIGH_RISK_REGIONS)
            | row["injury_region_2"].isin(_HIGH_RISK_REGIONS)
        ).astype(float),
    )
    row["multiple_injury_sites"] = np.where(
        row["injury_region_2"].isna(),
        np.nan,
        (~row["injury_region_2"].eq("none")).astype(float),
    )
    row["night_collision"] = row["Time of Collision"].map(
        lambda value: _optional_membership(value, _NIGHT_WINDOWS)
    )
    row["vulnerable_road_user"] = row["Mode Of Travel During Accident"].map(
        lambda value: _optional_membership(value, _VULNERABLE_USERS)
    )
    row["heavy_counterpart"] = row["Collision With "].map(
        lambda value: _optional_membership(value, _HEAVY_COUNTERPARTS)
    )
    row["ambulance_arrival"] = row["Mode of Transport to the Hospital"].map(
        lambda value: np.nan if _missing(value) else float(str(value) == "ambulance")
    )
    return row


def predict_severity(
    *,
    record: Optional[dict[str, Any]] = None,
    age: Optional[str] = None,
    mode_of_transport: Optional[str] = None,
    time_to_hospital: Optional[str] = None,
    collision_with: Optional[str] = None,
    site_of_injury: Optional[str] = None,
) -> ModelPrediction:
    contract, contract_error = _read_contract("severity")
    if contract_error or contract is None:
        return _severity_defer(
            "severity_triage",
            None,
            "UNAVAILABLE",
            contract_error or "Severity contract unavailable.",
        )

    model_name = str(contract.get("name", "severity_triage"))
    version = str(contract.get("version")) if contract.get("version") else None
    supplied = {
        "Age": age,
        "Mode of Transport to the Hospital": mode_of_transport,
        "Time Taken To Reach Hospital": time_to_hospital,
        "Collision With ": collision_with,
        "Site of Injury No1": site_of_injury,
    }
    if record:
        supplied.update(record)
    supplied = normalise_record(supplied)

    loaded, load_error = _load_bundle_model("severity", contract)
    if load_error or not isinstance(loaded, dict):
        return _severity_defer(
            model_name,
            version,
            "UNAVAILABLE",
            load_error or "Severity artifact is not a notebook bundle.",
        )

    raw_inputs = list(loaded.get("raw_inputs") or [])
    features = list(loaded.get("features") or [])
    categorical = list(loaded.get("cat_features") or [])
    if not raw_inputs or not features or "model" not in loaded:
        return _severity_defer(
            model_name, version, "UNAVAILABLE", "Severity notebook bundle is incomplete."
        )

    raw_record = {feature: supplied.get(feature) for feature in raw_inputs}
    present_count = sum(not _missing(value) for value in raw_record.values())
    completeness = present_count / len(raw_inputs)
    # The notebook measures this floor on validation data and stores it in the
    # bundle (2 of 12 raw inputs for the deployed model). Completeness counts only
    # values actually recorded; build_model_record no longer fills defaults.
    completeness_floor = float(loaded.get("completeness_floor", 2 / 12))

    if completeness < completeness_floor:
        return _severity_defer(
            model_name,
            version,
            "REFUSED",
            (
                f"Only {present_count} of {len(raw_inputs)} triage details "
                f"are available. Defer to clinical judgement."
            ),
            completeness,
        )

    try:
        row = _engineer_severity_row(pd.DataFrame([raw_record]))
        for feature, order in (loaded.get("ordinal_orders") or {}).items():
            if feature in row:
                row[feature] = row[feature].map({value: index for index, value in enumerate(order)})
        row = row[features]
        for feature in categorical:
            if feature in row:
                row[feature] = row[feature].astype(object)

        model = loaded["model"]
        raw_probabilities = np.asarray(model.predict_proba(row))[0]
        classes = [str(value) for value in model.classes_]
        positive_index = classes.index("1") if "1" in classes else len(classes) - 1
        positive_probability = float(raw_probabilities[positive_index])
        probabilities = {
            "No escalated care": round(float(raw_probabilities[1 - positive_index]), 4),
            "Escalated care": round(positive_probability, 4),
        }
        low_cut = float(loaded.get("low_cut", 0.08))
        high_cut = float(loaded.get("high_cut", 0.32))
        if positive_probability >= high_cut:
            prediction = SEVERITY_HIGH
            status = "OK"
            confidence_level = "high"
        elif positive_probability < low_cut:
            prediction = SEVERITY_LOW
            status = "OK"
            confidence_level = "high"
        else:
            prediction = SEVERITY_BORDERLINE
            status = "LOW_CONFIDENCE"
            confidence_level = "low"
    except Exception:
        # Details go to the server log; the stored message is shown to staff.
        logger.exception("Severity inference failed")
        return _severity_defer(
            model_name,
            version,
            "ERROR",
            "Severity inference failed. The details are in the server log.",
            completeness,
        )

    return ModelPrediction(
        prediction=prediction,
        probability=round(positive_probability, 4),
        probabilities=probabilities,
        completeness=round(completeness, 3),
        status=status,
        confidence_level=confidence_level,
        model_name=model_name,
        model_version=version,
        message=(
            f"{positive_probability:.0%} estimated chance of needing escalated care; "
            f"triage zone: {prediction}."
        ),
    )


def severity_readiness(record: dict[str, Any]) -> dict[str, Any]:
    """Say whether predict_severity would score this record, without running the model.

    Applies the same rule predict_severity uses before inference: the share of
    the bundle's raw inputs actually recorded must reach the bundle's
    completeness floor. When the bundle cannot be read, ``available`` is False
    and ``ready`` stays True so that a run reports the failure (a DEFER with
    status UNAVAILABLE) instead of silently waiting forever.
    """
    contract, contract_error = _read_contract("severity")
    loaded, load_error = (
        _load_bundle_model("severity", contract) if contract else (None, contract_error)
    )
    raw_inputs = list(loaded.get("raw_inputs") or []) if isinstance(loaded, dict) else []
    if load_error or not raw_inputs:
        return {
            "available": False,
            "ready": True,
            "raw_inputs": [],
            "present": [],
            "missing": [],
            "minimum": None,
            "message": load_error or "Severity notebook bundle is incomplete.",
        }

    supplied = normalise_record(record)
    present = [feature for feature in raw_inputs if not _missing(supplied.get(feature))]
    floor = float(loaded.get("completeness_floor", 2 / 12))
    minimum = next(
        (count for count in range(len(raw_inputs) + 1) if count / len(raw_inputs) >= floor),
        len(raw_inputs),
    )
    return {
        "available": True,
        "ready": len(present) >= minimum,
        "raw_inputs": raw_inputs,
        "present": present,
        "missing": [feature for feature in raw_inputs if feature not in present],
        "minimum": minimum,
        "message": None,
    }


# ---------------------------------------------------------------------------
# Contract-driven classification models
# ---------------------------------------------------------------------------

_bundle_cache: dict[str, tuple[float, Any]] = {}


def get_model_root() -> Path:
    """Return the directory from which all model artifacts are loaded."""
    return MODEL_ROOT


def get_model_source() -> str:
    """Return a safe, displayable description of the active model source."""
    return MODEL_SOURCE


def sync_models_from_huggingface(
    repo_id: Optional[str],
    *,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> tuple[bool, str]:
    """Download a Hub model bundle, then execute it from the local cache.

    A failed or absent Hub configuration deliberately leaves the packaged local
    model bundle active, so a temporary Hub outage never prevents the API from
    starting when local artifacts are available. ``revision`` (a commit hash
    or tag) pins the download so every start runs the same artifacts.
    """
    global MODEL_ROOT, MODEL_SOURCE

    if not repo_id:
        return False, "HF_MODEL_REPO is not configured; using local model bundle."

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False, "huggingface_hub is not installed; using local model bundle."

    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=token or None,
            local_dir=cache_dir or "/tmp/accident-care-models",
            revision=revision or None,
        )
    except Exception as exc:
        logger.warning("Hugging Face download failed: %s: %s", type(exc).__name__, exc)
        return False, "Could not download the Hugging Face model bundle; using the local bundle."

    downloaded_root = Path(downloaded_path)
    if not (downloaded_root / "severity").exists():
        return False, (
            "Hugging Face model bundle does not contain the expected top-level "
            "model folders; using local model bundle."
        )

    MODEL_ROOT = downloaded_root
    MODEL_SOURCE = f"Hugging Face model repository: {repo_id}" + (f"@{revision}" if revision else "")
    _bundle_cache.clear()
    return True, f"Loaded model bundle from Hugging Face repository {repo_id}" + (
        f" at revision {revision}." if revision else " (latest revision; set HF_MODEL_REVISION to pin it).")


def _read_contract(bundle_name: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    contract_path = MODEL_ROOT / bundle_name / "model_contract.json"
    if not contract_path.exists():
        # Messages name files relative to the model root: they reach API clients.
        return None, f"Contract is missing: {bundle_name}/model_contract.json"
    try:
        return json.loads(contract_path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s/%s: %s", bundle_name, contract_path.name, exc)
        return None, f"Could not read {bundle_name}/{contract_path.name}."


def _load_bundle_model(bundle_name: str, contract: dict[str, Any]) -> tuple[Optional[Any], Optional[str]]:
    artifact_name = contract.get("artifact")
    if not artifact_name:
        return None, "The model contract does not name an artifact."
    artifact_path = MODEL_ROOT / bundle_name / str(artifact_name)
    if not artifact_path.exists():
        return None, f"Artifact is not installed: {bundle_name}/{artifact_name}"

    modified = artifact_path.stat().st_mtime
    cached = _bundle_cache.get(str(artifact_path))
    if cached and cached[0] == modified:
        return cached[1], None

    try:
        artifact_format = str(contract.get("format", "joblib")).lower()
        if artifact_format == "catboost":
            model = CatBoostClassifier()
            model.load_model(str(artifact_path))
        elif artifact_format in {"joblib", "joblib_bundle"}:
            import joblib

            model = joblib.load(artifact_path)
        else:
            return None, f"Unsupported classification artifact format: {artifact_format}"
    except Exception as exc:
        logger.warning("Could not load %s/%s: %s", bundle_name, artifact_path.name, exc)
        return None, f"Could not load {bundle_name}/{artifact_path.name}."

    _bundle_cache[str(artifact_path)] = (modified, model)
    return model, None


def _bundle_settings(
    bundle_name: str,
    contract: dict[str, Any],
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    settings = dict(contract)
    settings["required_inputs"] = list(
        contract.get("required_inputs") or REQUIRED_INPUTS.get(bundle_name, ())
    )
    metadata_name = contract.get("metadata_artifact")
    if not metadata_name:
        return settings, None

    metadata_path = MODEL_ROOT / bundle_name / str(metadata_name)
    if not metadata_path.exists():
        return None, f"Metadata artifact is not installed: {bundle_name}/{metadata_name}"

    modified = metadata_path.stat().st_mtime
    cached = _bundle_cache.get(str(metadata_path))
    if cached and cached[0] == modified:
        metadata = cached[1]
    else:
        try:
            import joblib

            metadata = joblib.load(metadata_path)
        except Exception as exc:
            logger.warning("Could not load %s/%s: %s", bundle_name, metadata_path.name, exc)
            return None, f"Could not load {bundle_name}/{metadata_path.name}."
        _bundle_cache[str(metadata_path)] = (modified, metadata)

    if not isinstance(metadata, dict):
        return None, f"{metadata_path.name} must contain a metadata dictionary."

    settings["features"] = list(metadata.get("features") or settings.get("features") or [])
    settings["categorical_features"] = list(
        metadata.get("cat_features") or settings.get("categorical_features") or []
    )
    settings["numeric_features"] = list(
        metadata.get("num_features") or settings.get("numeric_features") or []
    )
    settings["rare_maps"] = metadata.get("rare_maps") or {}
    if metadata.get("completeness_floor") is not None:
        settings["completeness_floor"] = float(metadata["completeness_floor"])
    if metadata.get("low_conf_prob") is not None:
        settings["confidence_floor"] = float(metadata["low_conf_prob"])
    if metadata.get("chosen_threshold") is not None:
        settings["decision_threshold"] = float(metadata["chosen_threshold"])

    ordinal_mappings = dict(settings.get("ordinal_mappings") or {})
    if metadata.get("age_order"):
        ordinal_mappings["Age"] = {
            value: index for index, value in enumerate(metadata["age_order"])
        }
    if metadata.get("time_order"):
        ordinal_mappings["Time Taken To Reach Hospital"] = {
            value: index for index, value in enumerate(metadata["time_order"])
        }
    settings["ordinal_mappings"] = ordinal_mappings
    return settings, None


def _prepare_bundle_row(
    record: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[Optional[pd.DataFrame], float, Optional[str]]:
    features = settings.get("features") or []
    if not features:
        return None, 0, "The model contract has no features."

    record = normalise_record(record)
    raw = {feature: record.get(feature) for feature in features}
    completeness = sum(not _missing(value) for value in raw.values()) / len(features)
    awaiting = missing_required_inputs(record, settings)
    if awaiting:
        return None, completeness, (
            "Waiting for post-treatment inputs: " + ", ".join(awaiting) + "."
        )
    floor = float(settings.get("completeness_floor", COMPLETENESS_FLOOR))
    if completeness < floor:
        return None, completeness, "Not enough information to run this model."

    row = pd.DataFrame([raw], columns=features)
    for feature, mapping in (settings.get("value_mappings") or {}).items():
        if feature in row:
            row[feature] = row[feature].map(
                lambda value: mapping.get(str(value), value) if not _missing(value) else value
            )
    for feature, keep_values in (settings.get("rare_maps") or {}).items():
        if feature in row:
            keep = set(keep_values)
            row[feature] = row[feature].map(
                lambda value: value if _missing(value) or value in keep else "other"
            )
    for feature, mapping in (settings.get("ordinal_mappings") or {}).items():
        if feature in row:
            row[feature] = row[feature].map(mapping)

    for feature in settings.get("numeric_features") or []:
        if feature in row:
            row[feature] = pd.to_numeric(row[feature], errors="coerce")

    artifact_format = str(settings.get("format", "joblib")).lower()
    for feature in settings.get("categorical_features") or []:
        if feature in row:
            if artifact_format == "catboost":
                row[feature] = (
                    row[feature].astype(object).where(row[feature].notna(), "missing").astype(str)
                )
            else:
                row[feature] = row[feature].astype(object).where(row[feature].notna(), np.nan)

    return row, completeness, None


def bundle_readiness(bundle_name: str, record: dict[str, Any]) -> dict[str, Any]:
    """Say whether predict_bundle would score this record, without running the model.

    Applies the same two checks as _prepare_bundle_row: every required
    (post-treatment) input is recorded, and the share of recorded features
    reaches the bundle's completeness floor. When the bundle cannot be read,
    ``available`` is False and ``ready`` stays True so a run reports the failure.
    """
    contract, contract_error = _read_contract(bundle_name)
    settings, settings_error = (
        _bundle_settings(bundle_name, contract) if contract else (None, contract_error)
    )
    features = list((settings or {}).get("features") or [])
    if settings_error or settings is None or not features:
        return {
            "available": False,
            "ready": True,
            "missing_required": [],
            "missing_features": [],
            "completeness": None,
            "minimum_completeness": None,
            "message": settings_error or "The model contract has no features.",
        }

    supplied = normalise_record(record)
    missing_features = [feature for feature in features if _missing(supplied.get(feature))]
    completeness = (len(features) - len(missing_features)) / len(features)
    missing_required = missing_required_inputs(supplied, settings)
    floor = float(settings.get("completeness_floor", COMPLETENESS_FLOOR))
    return {
        "available": True,
        "ready": not missing_required and completeness >= floor,
        "missing_required": missing_required,
        "missing_features": missing_features,
        "completeness": round(completeness, 3),
        "minimum_completeness": round(floor, 3),
        "message": None,
    }


def predict_bundle(bundle_name: str, record: dict[str, Any]) -> ModelPrediction:
    contract, contract_error = _read_contract(bundle_name)
    if contract_error or contract is None:
        return _unavailable(bundle_name, None, contract_error or "Model contract unavailable.")

    model_name = str(contract.get("name", bundle_name))
    version = str(contract.get("version")) if contract.get("version") else None
    settings, settings_error = _bundle_settings(bundle_name, contract)
    if settings_error or settings is None:
        return _unavailable(model_name, version, settings_error or "Model metadata unavailable.")

    row, completeness, row_error = _prepare_bundle_row(record, settings)
    if row_error:
        return ModelPrediction(
            prediction=None,
            probability=None,
            probabilities={},
            completeness=round(completeness, 3),
            status="REFUSED" if settings.get("features") else "UNAVAILABLE",
            confidence_level=None,
            model_name=model_name,
            model_version=version,
            message=row_error,
        )

    model, load_error = _load_bundle_model(bundle_name, contract)
    if load_error or model is None:
        result = _unavailable(model_name, version, load_error or "Model artifact unavailable.")
        result.completeness = round(completeness, 3)
        return result

    try:
        raw_probabilities = np.asarray(model.predict_proba(row))[0]
        raw_classes = list(model.classes_)
        label_map = {str(key): str(value) for key, value in (settings.get("label_map") or {}).items()}
        display_classes = [label_map.get(str(value), str(value)) for value in raw_classes]
        probabilities = {
            label: round(float(probability), 4)
            for label, probability in zip(display_classes, raw_probabilities)
        }

        positive_class = settings.get("positive_class")
        threshold = settings.get("decision_threshold")
        threshold_mode = positive_class is not None and threshold is not None
        if threshold_mode:
            positive_index = next(
                index for index, value in enumerate(raw_classes)
                if str(value) == str(positive_class)
            )
            positive_probability = float(raw_probabilities[positive_index])
            negative_index = 1 - positive_index if len(raw_classes) == 2 else int(np.argmax(raw_probabilities))
            selected_index = positive_index if positive_probability >= float(threshold) else negative_index
        else:
            selected_index = int(np.argmax(raw_probabilities))

        prediction = display_classes[selected_index]
        probability = float(raw_probabilities[selected_index])
    except Exception:
        logger.exception("%s inference failed", model_name)
        return ModelPrediction(
            prediction=None,
            probability=None,
            probabilities={},
            completeness=round(completeness, 3),
            status="ERROR",
            confidence_level=None,
            model_name=model_name,
            model_version=version,
            message="Inference failed. The details are in the server log.",
        )

    if threshold_mode:
        # Mirrors the transfer notebook: the reported probability is the event
        # probability, the label comes from the tuned threshold, and the
        # notebook defines no low-confidence band for this decision rule.
        positive_label = display_classes[positive_index]
        return ModelPrediction(
            prediction=prediction,
            probability=round(positive_probability, 4),
            probabilities=probabilities,
            completeness=round(completeness, 3),
            status="OK",
            confidence_level=None,
            model_name=model_name,
            model_version=version,
            message=(
                f"{positive_probability:.0%} estimated probability of {positive_label.lower()}; "
                f"labelled {positive_label.lower()} at {float(threshold):.0%} or higher."
            ),
        )

    confidence_floor = float(settings.get("confidence_floor", LOW_CONF_PROB))
    confidence_level = "high" if probability >= confidence_floor else "low"
    return ModelPrediction(
        prediction=prediction,
        probability=round(probability, 4),
        probabilities=probabilities,
        completeness=round(completeness, 3),
        status="OK" if confidence_level == "high" else "LOW_CONFIDENCE",
        confidence_level=confidence_level,
        model_name=model_name,
        model_version=version,
    )


def predict_stay_duration(
    record: Optional[dict[str, Any]] = None,
    *,
    time_to_hospital: Optional[str] = None,
    collision_force_from: Optional[str] = None,
    site_of_injury: Optional[str] = None,
) -> ModelPrediction:
    supplied = dict(record or {})
    supplied.setdefault("Time Taken To Reach Hospital", time_to_hospital)
    supplied.setdefault("Collision Force From", collision_force_from)
    supplied.setdefault("Site of Injury No1", site_of_injury)
    return predict_bundle("stay_duration", supplied)


def predict_transfer_probability(
    record: Optional[dict[str, Any]] = None,
    *,
    age: Optional[str] = None,
    mode_of_transport: Optional[str] = None,
    time_to_hospital: Optional[str] = None,
    collision_with: Optional[str] = None,
    speed: Optional[str] = None,
    road_type: Optional[str] = None,
    site_of_injury: Optional[str] = None,
) -> ModelPrediction:
    supplied = dict(record or {})
    supplied.setdefault("Age", age)
    supplied.setdefault("Mode of Transport to the Hospital", mode_of_transport)
    supplied.setdefault("Time Taken To Reach Hospital", time_to_hospital)
    supplied.setdefault("Collision With", collision_with)
    supplied.setdefault("Approximate Speed", speed)
    supplied.setdefault("Road Type", road_type)
    supplied.setdefault("Site of Injury No1", site_of_injury)
    return predict_bundle("transfer", supplied)


def predict_discharge_outcome(record: dict[str, Any]) -> ModelPrediction:
    return predict_bundle("discharge", record)


# ---------------------------------------------------------------------------
# Forecast model
# ---------------------------------------------------------------------------

def predict_accident_forecast(horizon: int = 6) -> ForecastPrediction:
    contract, contract_error = _read_contract("forecasting")
    if contract_error or contract is None:
        return ForecastPrediction(
            status="UNAVAILABLE",
            model_name="accident_forecast",
            model_version=None,
            points=[],
            message=contract_error,
        )

    model_name = str(contract.get("name", "accident_forecast"))
    version = str(contract.get("version")) if contract.get("version") else None
    artifact_name = contract.get("artifact")
    artifact_path = MODEL_ROOT / "forecasting" / str(artifact_name)
    if not artifact_name or not artifact_path.exists():
        return ForecastPrediction(
            status="UNAVAILABLE",
            model_name=model_name,
            model_version=version,
            points=[],
            message=f"Forecast artifact is not installed: forecasting/{artifact_name}",
        )

    artifact_format = str(contract.get("format", "statsmodels")).lower()

    try:
        if artifact_format == "seasonal_naive_json":
            bundle = json.loads(artifact_path.read_text(encoding="utf-8"))
            seasonal_values = bundle.get("seasonal_values") or []
            season_length = int(bundle.get("season_length", 12))
            if len(seasonal_values) < season_length:
                raise ValueError(
                    f"Seasonal-naive artifact needs {season_length} values; "
                    f"found {len(seasonal_values)}."
                )
            last_period = pd.Timestamp(bundle["last_observed_period"])
            source_periods = bundle.get("source_periods")
            if source_periods:
                if len(source_periods) < season_length:
                    raise ValueError(
                        f"Seasonal-naive artifact needs {season_length} source periods; "
                        f"found {len(source_periods)}."
                    )
                seasonal_periods = [
                    pd.Timestamp(period)
                    for period in source_periods[-season_length:]
                ]
            else:
                seasonal_periods = list(
                    pd.date_range(end=last_period, periods=season_length, freq="MS")
                )

            seasonal_by_month = {
                period.month: float(value)
                for period, value in zip(
                    seasonal_periods,
                    seasonal_values[-season_length:],
                )
            }
            if len(seasonal_by_month) != season_length:
                raise ValueError(
                    "Seasonal-naive source periods must contain one value for each "
                    "calendar month."
                )

            current_period = (
                pd.Timestamp.now(tz="UTC")
                .tz_localize(None)
                .to_period("M")
                .to_timestamp()
            )
            future_periods = pd.date_range(
                current_period + pd.offsets.MonthBegin(1),
                periods=horizon,
                freq="MS",
            )
            risk_thresholds = _forecast_risk_thresholds(
                [float(value) for value in seasonal_values[-season_length:]]
            )
            points = [
                ForecastPoint(
                    period=period.strftime("%b %Y"),
                    predicted_accidents=round(
                        max(0.0, seasonal_by_month[period.month]),
                        1,
                    ),
                    risk_level=_forecast_risk_level(
                        max(0.0, seasonal_by_month[period.month]),
                        risk_thresholds,
                    ),
                )
                for period in future_periods
            ]
            return ForecastPrediction(
                status="OK",
                model_name=model_name,
                model_version=version,
                points=points,
                message=(
                    "Seasonal-naive forecasts repeat each corresponding calendar-month "
                    f"value from the artifact's annual cycle ending {last_period:%b %Y}. "
                    "Retrain the artifact with recent data before treating these values "
                    "as current-scale forecasts."
                ),
                risk_thresholds=risk_thresholds,
            )

        if artifact_format != "statsmodels":
            raise ValueError(f"Unsupported forecast artifact format: {artifact_format}")

        modified = artifact_path.stat().st_mtime
        cached = _bundle_cache.get(str(artifact_path))
        if cached and cached[0] == modified:
            model = cached[1]
        else:
            from statsmodels.iolib.smpickle import load_pickle

            model = load_pickle(artifact_path)
            _bundle_cache[str(artifact_path)] = (modified, model)

        forecast = model.get_forecast(steps=horizon)
        predicted = forecast.predicted_mean
        confidence = forecast.conf_int()
        points: list[ForecastPoint] = []
        for index, value in enumerate(np.asarray(predicted).reshape(-1)):
            label = predicted.index[index] if hasattr(predicted, "index") else index + 1
            period = label.strftime("%b %Y") if hasattr(label, "strftime") else str(label)
            interval = np.asarray(
                confidence.iloc[index] if hasattr(confidence, "iloc") else confidence[index]
            ).reshape(-1)
            points.append(ForecastPoint(
                period=period,
                predicted_accidents=round(max(0.0, float(value)), 1),
                lower_bound=round(max(0.0, float(interval[0])), 1) if len(interval) >= 2 else None,
                upper_bound=round(max(0.0, float(interval[1])), 1) if len(interval) >= 2 else None,
            ))
    except Exception:
        logger.exception("Forecast inference failed")
        return ForecastPrediction(
            status="ERROR",
            model_name=model_name,
            model_version=version,
            points=[],
            message="Forecast inference failed. The details are in the server log.",
        )

    return ForecastPrediction(
        status="OK",
        model_name=model_name,
        model_version=version,
        points=points,
    )


# ---------------------------------------------------------------------------
# Version identity and input fields (used to persist predictions)
# ---------------------------------------------------------------------------

_file_digest_cache: dict[str, tuple[float, int, str]] = {}


def _file_sha256(path: Path) -> str:
    stat = path.stat()
    cached = _file_digest_cache.get(str(path))
    if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return cached[2]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    _file_digest_cache[str(path)] = (stat.st_mtime, stat.st_size, digest.hexdigest())
    return digest.hexdigest()


def model_fingerprint(bundle_name: str) -> Optional[dict[str, Any]]:
    """Identify the exact model a prediction came from.

    The fingerprint is a SHA-256 over the model contract, the artifact and
    metadata files, and this module's input gate for the bundle, so any change
    to what produces a prediction yields a new identity. The contract's own
    "version" string is kept alongside but is not unique per artifact.
    """
    contract, error = _read_contract(bundle_name)
    if error or contract is None:
        return None
    digest = hashlib.sha256()
    digest.update(json.dumps(contract, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(list(REQUIRED_INPUTS.get(bundle_name, ()))).encode("utf-8"))
    for key in ("artifact", "metadata_artifact"):
        name = contract.get(key)
        if not name:
            continue
        path = MODEL_ROOT / bundle_name / str(name)
        digest.update(f"{key}={name}:".encode("utf-8"))
        digest.update((_file_sha256(path) if path.exists() else "missing").encode("utf-8"))
    return {
        "model_key": bundle_name,
        "fingerprint": digest.hexdigest(),
        "contract_name": contract.get("name"),
        "contract_version": contract.get("version"),
        "artifact_name": str(contract.get("artifact") or "unknown"),
        "source": MODEL_SOURCE,
    }


def model_input_fields(bundle_name: str) -> Optional[list[str]]:
    """Return the raw input fields a model reads, or None if it cannot be loaded."""
    contract, error = _read_contract(bundle_name)
    if error or contract is None:
        return None
    if bundle_name == "severity":
        loaded, load_error = _load_bundle_model(bundle_name, contract)
        if load_error or not isinstance(loaded, dict):
            return None
        return list(loaded.get("raw_inputs") or [])
    settings, settings_error = _bundle_settings(bundle_name, contract)
    if settings_error or settings is None:
        return None
    return list(settings.get("features") or [])


def get_model_status() -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for bundle_name in ("severity", "stay_duration", "transfer", "discharge", "forecasting"):
        contract, error = _read_contract(bundle_name)
        artifact = None
        available = False
        if contract and contract.get("artifact"):
            artifact = MODEL_ROOT / bundle_name / str(contract["artifact"])
            available = artifact.exists()
        load_error = None
        if available and contract:
            if bundle_name == "forecasting":
                forecast_result = predict_accident_forecast(1)
                if forecast_result.status != "OK":
                    load_error = forecast_result.message or forecast_result.status
            else:
                loaded, load_error = _load_bundle_model(bundle_name, contract)
                if not load_error and bundle_name == "severity" and not isinstance(loaded, dict):
                    load_error = "Severity artifact is not a notebook bundle."
                if not load_error:
                    _, load_error = _bundle_settings(bundle_name, contract)
        if error or load_error:
            logger.warning("Model bundle %s not ready: %s", bundle_name, error or load_error)
        # Public health output: readiness only, no file system paths or error internals.
        status[bundle_name] = {
            "available": available,
            "loadable": available and load_error is None,
            "artifact": artifact.name if artifact else None,
            "version": contract.get("version") if contract else None,
            "message": (
                "Model contract could not be read." if error
                else "Artifact could not be loaded." if load_error
                else None if available else "Artifact not installed."
            ),
        }
    return status
