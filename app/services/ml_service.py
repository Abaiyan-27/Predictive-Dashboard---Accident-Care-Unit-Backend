"""Model loading and inference for every prediction shown by the dashboard.

Each deployed notebook artifact has a small ``model_contract.json`` beside it.
Missing artifacts, incomplete inputs, and inference failures are reported
explicitly; the service never substitutes demo or randomly generated values.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


MODEL_ROOT = Path(__file__).resolve().parent.parent / "ml_models"

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
        return _unavailable("severity_triage", None, contract_error or "Severity contract unavailable.")

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

    loaded, load_error = _load_bundle_model("severity", contract)
    if load_error or not isinstance(loaded, dict):
        return _unavailable(
            model_name,
            version,
            load_error or "Severity artifact is not a notebook bundle.",
        )

    raw_inputs = list(loaded.get("raw_inputs") or [])
    features = list(loaded.get("features") or [])
    categorical = list(loaded.get("cat_features") or [])
    if not raw_inputs or not features or "model" not in loaded:
        return _unavailable(model_name, version, "Severity notebook bundle is incomplete.")

    raw_record = {feature: supplied.get(feature) for feature in raw_inputs}
    completeness = sum(not _missing(value) for value in raw_record.values()) / len(raw_inputs)
    completeness_floor = float(loaded.get("completeness_floor", COMPLETENESS_FLOOR))

    if completeness < completeness_floor:
        return ModelPrediction(
            prediction=None,
            probability=None,
            probabilities={},
            completeness=round(completeness, 3),
            status="REFUSED",
            confidence_level=None,
            model_name=model_name,
            model_version=version,
            message=(
                f"Only {sum(not _missing(value) for value in raw_record.values())} of "
                f"{len(raw_inputs)} triage details are available."
            ),
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
            prediction = "High"
            status = "OK"
            confidence_level = "high"
        elif positive_probability < low_cut:
            prediction = "Low"
            status = "OK"
            confidence_level = "high"
        else:
            prediction = "Borderline"
            status = "LOW_CONFIDENCE"
            confidence_level = "low"
    except Exception as exc:
        return ModelPrediction(
            prediction=None,
            probability=None,
            probabilities={},
            completeness=round(completeness, 3),
            status="ERROR",
            confidence_level=None,
            model_name=model_name,
            model_version=version,
            message=f"Severity inference failed: {exc}",
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


# ---------------------------------------------------------------------------
# Contract-driven classification models
# ---------------------------------------------------------------------------

_bundle_cache: dict[str, tuple[float, Any]] = {}


def _read_contract(bundle_name: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    contract_path = MODEL_ROOT / bundle_name / "model_contract.json"
    if not contract_path.exists():
        return None, f"Contract is missing: {contract_path}"
    try:
        return json.loads(contract_path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Could not read {contract_path.name}: {exc}"


def _load_bundle_model(bundle_name: str, contract: dict[str, Any]) -> tuple[Optional[Any], Optional[str]]:
    artifact_name = contract.get("artifact")
    if not artifact_name:
        return None, "The model contract does not name an artifact."
    artifact_path = MODEL_ROOT / bundle_name / str(artifact_name)
    if not artifact_path.exists():
        return None, f"Artifact is not installed: {artifact_path}"

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
        return None, f"Could not load {artifact_path.name}: {exc}"

    _bundle_cache[str(artifact_path)] = (modified, model)
    return model, None


def _bundle_settings(
    bundle_name: str,
    contract: dict[str, Any],
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    settings = dict(contract)
    metadata_name = contract.get("metadata_artifact")
    if not metadata_name:
        return settings, None

    metadata_path = MODEL_ROOT / bundle_name / str(metadata_name)
    if not metadata_path.exists():
        return None, f"Metadata artifact is not installed: {metadata_path}"

    modified = metadata_path.stat().st_mtime
    cached = _bundle_cache.get(str(metadata_path))
    if cached and cached[0] == modified:
        metadata = cached[1]
    else:
        try:
            import joblib

            metadata = joblib.load(metadata_path)
        except Exception as exc:
            return None, f"Could not load {metadata_path.name}: {exc}"
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

    raw = {feature: record.get(feature) for feature in features}
    completeness = sum(not _missing(value) for value in raw.values()) / len(features)
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
        if positive_class is not None and threshold is not None:
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
    except Exception as exc:
        return ModelPrediction(
            prediction=None,
            probability=None,
            probabilities={},
            completeness=round(completeness, 3),
            status="ERROR",
            confidence_level=None,
            model_name=model_name,
            model_version=version,
            message=f"Inference failed: {exc}",
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
            message=f"Forecast artifact is not installed: {artifact_path}",
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
    except Exception as exc:
        return ForecastPrediction(
            status="ERROR",
            model_name=model_name,
            model_version=version,
            points=[],
            message=f"Forecast inference failed: {exc}",
        )

    return ForecastPrediction(
        status="OK",
        model_name=model_name,
        model_version=version,
        points=points,
    )


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
        status[bundle_name] = {
            "available": available,
            "loadable": available and load_error is None,
            "artifact": str(artifact) if artifact else None,
            "version": contract.get("version") if contract else None,
            "message": error if error else (
                load_error if load_error else (None if available else "Artifact not installed.")
            ),
        }
    return status
