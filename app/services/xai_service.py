import logging
from pathlib import Path
from typing import Any, Optional, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import entropy

from app.services.ml_service import _read_contract, _bundle_settings, _load_bundle_model, MODEL_ROOT, _prepare_bundle_row

logger = logging.getLogger("xai_engine")

DEFAULT_K_NEIGHBORS = 30
DEFAULT_MAX_CANDIDATES = 4
EPISTEMIC_THRESHOLD_BITS = 0.15

SUPPORTED_MODELS = ("discharge", "transfer", "stay_duration")

class XAIEngineError(Exception):
    """Raised for any XAI engine failure that should surface as a clean API error."""


class UnknownModelError(XAIEngineError):
    """Requested model_key is not one of the supported models."""


class ModelLoadError(XAIEngineError):
    """A required model or metadata file could not be loaded."""

# Global state for XAI
_MODELS: Dict[str, Any] = {}
_TRAIN_POOLS: Dict[str, pd.DataFrame] = {}

def get_xai_models() -> Dict[str, Any]:
    global _MODELS
    if _MODELS:
        return _MODELS

    for key in SUPPORTED_MODELS:
        contract, contract_error = _read_contract(key)
        if contract_error or contract is None:
            raise ModelLoadError(f"Missing contract for {key}")
        settings, settings_error = _bundle_settings(key, contract)
        if settings_error or settings is None:
            raise ModelLoadError(f"Missing metadata for {key}")
        model, load_error = _load_bundle_model(key, contract)
        if load_error or model is None:
            raise ModelLoadError(f"Missing model for {key}")
        
        # Ensure age_order and time_order exist
        if "age_order" not in settings and "Age" in settings.get("ordinal_mappings", {}):
            # Reverse engineer age order from mapping
            mapping = settings["ordinal_mappings"]["Age"]
            order = [""] * len(mapping)
            for k, v in mapping.items():
                if v < len(order):
                    order[v] = k
            settings["age_order"] = order

        if "time_order" not in settings and "Time Taken To Reach Hospital" in settings.get("ordinal_mappings", {}):
            mapping = settings["ordinal_mappings"]["Time Taken To Reach Hospital"]
            order = [""] * len(mapping)
            for k, v in mapping.items():
                if v < len(order):
                    order[v] = k
            settings["time_order"] = order

        _MODELS[key] = {"model": model, **settings}
    return _MODELS


def consolidate_rare(frame: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    for col, keep in meta.get("rare_maps", {}).items():
        if col in out.columns:
            keep_set = set(keep)
            out[col] = out[col].where(out[col].isna() | out[col].isin(keep_set), "other")
    return out


def load_training_pools(data_path: str = None) -> Dict[str, pd.DataFrame]:
    global _TRAIN_POOLS
    if _TRAIN_POOLS:
        return _TRAIN_POOLS

    if data_path is None:
        data_path = str(MODEL_ROOT / "Preprocessed_Dataset.csv")
    
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        raise ModelLoadError(f"Training data not found at {data_path_obj}")

    df_raw = pd.read_csv(data_path_obj)
    for c in df_raw.columns:
        if df_raw[c].dtype == object:
            df_raw[c] = (df_raw[c].astype(str).str.strip().str.lower()
                        .replace({"nan": np.nan, "none": np.nan}))
    
    if "Collision With " in df_raw.columns:
        df_raw = df_raw.rename(columns={"Collision With ": "Collision With"})

    ward_cols = ["Number of days in first ward", "Number of days in Second ward",
                "Number of days in Third ward"]
    missing_ward_cols = [c for c in ward_cols if c not in df_raw.columns]
    if missing_ward_cols:
        raise ModelLoadError(f"Expected ward-day columns not found: {missing_ward_cols}")
    
    for c in ward_cols:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
        reason_col = f"{c}_missing_reason"
        if reason_col in df_raw.columns:
            early = df_raw[reason_col] == "early_discharge"
            df_raw.loc[early, c] = df_raw.loc[early, c].fillna(0)
    df_raw["total_days"] = df_raw[ward_cols].sum(axis=1, min_count=1)

    models = get_xai_models()
    for key, meta in models.items():
        missing_features = [f for f in meta["features"] if f not in df_raw.columns]
        if missing_features:
            raise ModelLoadError(f"Model '{key}' expects missing features: {missing_features}")
        pool = df_raw[meta["features"]].copy()
        pool = consolidate_rare(pool, meta)
        _TRAIN_POOLS[key] = pool
    return _TRAIN_POOLS


def encode_ordinal(frame: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    if "age_order" in meta:
        col = "Age"
        if col in out.columns:
            out[col] = out[col].map({v: i for i, v in enumerate(meta["age_order"])})
    if "time_order" in meta:
        col = "Time Taken To Reach Hospital"
        if col in out.columns:
            out[col] = out[col].map({v: i for i, v in enumerate(meta["time_order"])})
    return out


def predict_proba_row(model_key: str, record: Dict[str, Any]) -> tuple[np.ndarray, List[str]]:
    models = get_xai_models()
    meta = models[model_key]
    
    # We use _prepare_bundle_row which handles everything instead of rewriting it
    # We pass it as dict, and get out a dataframe formatted correctly for the model
    row, completeness, row_error = _prepare_bundle_row(record, meta)
    if row_error:
        raise XAIEngineError(f"Model '{model_key}' failed to score the given record: {row_error}")

    try:
        proba = meta["model"].predict_proba(row)[0]
    except Exception as exc:
        raise XAIEngineError(f"Model '{model_key}' failed to score the given record: {exc}") from exc
    classes = list(meta["model"].classes_)
    return proba, classes


def entropy_bits(proba: np.ndarray) -> float:
    return float(entropy(proba, base=2))


def gower_distance(model_key: str, record: Dict[str, Any], exclude_field: Optional[str] = None):
    models = get_xai_models()
    pools = load_training_pools()
    
    meta = models[model_key]
    pool = pools[model_key]
    features = [f for f in meta["features"] if f != exclude_field]
    ord_cols = {}
    if "age_order" in meta:
        ord_cols["Age"] = meta["age_order"]
    if "time_order" in meta:
        ord_cols["Time Taken To Reach Hospital"] = meta["time_order"]

    n = len(pool)
    total_dist = np.zeros(n)
    total_weight = np.zeros(n)

    for f in features:
        rv = record.get(f)
        if rv is None or (isinstance(rv, float) and pd.isna(rv)):
            continue
        if f in ord_cols:
            order = ord_cols[f]
            order_map = {v: i for i, v in enumerate(order)}
            if rv not in order_map:
                continue
            rv_idx = order_map[rv]
            pool_idx = pool[f].map(order_map)
            known = pool_idx.notna().values
            d = (pool_idx - rv_idx).abs().values / (len(order) - 1)
            total_dist[known] += np.nan_to_num(d[known])
            total_weight[known] += 1.0
        else:
            pool_vals = pool[f].values
            known = pd.notna(pool_vals)
            d = np.where(pool_vals == rv, 0.0, 1.0)
            total_dist[known] += d[known]
            total_weight[known] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(total_weight > 0, total_dist / np.maximum(total_weight, 1), np.inf)
    return avg, total_weight


def plausible_values(model_key: str, record: Dict[str, Any], field: str, k: int = DEFAULT_K_NEIGHBORS,
                     max_candidates: int = DEFAULT_MAX_CANDIDATES, exclude_current: bool = False):
    d, w = gower_distance(model_key, record, exclude_field=field)
    valid = w >= 1
    pools = load_training_pools()
    pool = pools[model_key]
    idx_sorted = np.argsort(np.where(valid, d, np.inf))
    neighbor_idx = idx_sorted[:k]
    neighbor_vals = pool[field].iloc[neighbor_idx].dropna()

    if exclude_current:
        current = record.get(field)
        neighbor_vals = neighbor_vals[neighbor_vals != current]

    if len(neighbor_vals) == 0:
        return []

    counts = neighbor_vals.value_counts(normalize=True)
    top = counts.head(max_candidates)
    return list(zip(top.index.tolist(), top.values.tolist()))


def simulate_missing_field(model_key: str, record: Dict[str, Any], field: str, k: int = DEFAULT_K_NEIGHBORS,
                           max_candidates: int = DEFAULT_MAX_CANDIDATES):
    current_proba, _ = predict_proba_row(model_key, record)
    current_entropy = entropy_bits(current_proba)

    candidates = plausible_values(model_key, record, field, k=k, max_candidates=max_candidates)
    if not candidates:
        return None

    sim_entropies, weights = [], []
    for value, weight in candidates:
        modified = dict(record)
        modified[field] = value
        proba, _ = predict_proba_row(model_key, modified)
        sim_entropies.append(entropy_bits(proba))
        weights.append(weight)

    weighted_avg_entropy = float(np.average(sim_entropies, weights=weights))
    return {"field": field, "current_entropy_bits": round(current_entropy, 4),
            "expected_post_entropy_bits": round(weighted_avg_entropy, 4),
            "expected_reduction_bits": round(current_entropy - weighted_avg_entropy, 4),
            "n_candidates_simulated": len(candidates)}


def decision_from_proba(model_key: str, proba: np.ndarray, classes: List[str]) -> tuple[str, float, float]:
    """Apply the deployed model's decision rule, including transfer's threshold."""
    models = get_xai_models()
    if model_key == "transfer":
        yes_idx = classes.index("yes")
        event_probability = float(proba[yes_idx])
        threshold = float(models[model_key].get("decision_threshold", 0.5))
        prediction = "yes" if event_probability >= threshold else "no"
        prediction_idx = classes.index(prediction)
        return prediction, event_probability, float(proba[prediction_idx])

    prediction_idx = int(np.argmax(proba))
    prediction = classes[prediction_idx]
    prediction_probability = float(proba[prediction_idx])
    return prediction, prediction_probability, prediction_probability


def simulate_known_field(model_key: str, record: Dict[str, Any], field: str, k: int = DEFAULT_K_NEIGHBORS,
                         max_candidates: int = DEFAULT_MAX_CANDIDATES):
    current_proba, classes = predict_proba_row(model_key, record)
    pred_class, event_prob, pred_confidence = decision_from_proba(model_key, current_proba, classes)

    candidates = plausible_values(model_key, record, field, k=k, max_candidates=max_candidates,
                                  exclude_current=True)
    if not candidates:
        return None

    results = []
    for value, weight in candidates:
        modified = dict(record)
        modified[field] = value
        proba, _ = predict_proba_row(model_key, modified)
        new_class, _, new_confidence = decision_from_proba(model_key, proba, classes)
        confidence_drop = pred_confidence - new_confidence
        results.append({"alternative_value": value, "resulting_prediction": str(new_class),
                        "confidence_drop": round(confidence_drop, 4),
                        "confidence_shift_abs": round(abs(confidence_drop), 4),
                        "flips_prediction": new_class != pred_class})

    most_sensitive = max(results, key=lambda r: r["confidence_shift_abs"])

    return {"field": field, "current_prediction": pred_class,
            "current_probability": round(event_prob, 4), "alternatives": results,
            "max_confidence_shift": most_sensitive["confidence_shift_abs"],
            "most_sensitive_alternative": most_sensitive["alternative_value"],
            "any_flip": any(r["flips_prediction"] for r in results)}


def _confidence_label(probability: float) -> str:
    if probability >= 0.75:
        return "high"
    if probability >= 0.50:
        return "moderate"
    return "low"


def _field_label(field: str) -> str:
    labels = {
        "Site of Injury No1": "main injury site",
        "Site of injury No 2": "second injury site",
        "Type of injury No 1": "main injury type",
        "Type of Injury No 2": "second injury type",
        "Mode of Transport to the Hospital": "transport to hospital",
        "Time Taken To Reach Hospital": "time taken to reach hospital",
        "Transfer To Next Hospital": "transfer status",
        "Investigation Done": "investigation performed",
        "Management Name 1": "primary management",
        "Management Name 2": "secondary management",
    }
    return labels.get(field, field.lower())


def _prediction_label(model_key: str, prediction: str) -> str:
    if model_key == "transfer":
        return f"Prediction: Transfer is {'likely' if prediction == 'yes' else 'unlikely'}."
    if model_key == "stay":
        return f"Prediction: Expected hospital stay is {prediction}."
    return f"Prediction: Expected discharge outcome is {prediction}."


def _interpretation_line(model_key: str, pred_class: str) -> str:
    models = get_xai_models()
    if model_key == "transfer":
        threshold = models[model_key].get("decision_threshold", 0.5)
        return (f"Meaning: Transfer is labelled likely at an estimated transfer probability of "
                f"{float(threshold):.0%} or higher; this is a decision-support flag, not an instruction to transfer.")
    if model_key == "stay":
        return (f"Meaning: This is an estimate of the patient's stay-duration group ({pred_class}), "
                "not a guaranteed discharge date or bed requirement.")
    return (f"Meaning: This is an estimate of the recorded discharge-outcome group ({pred_class}), "
            "not a diagnosis or a replacement for clinical assessment.")


def explain(model_key: str, record: Dict[str, Any], k: int = DEFAULT_K_NEIGHBORS, 
            max_candidates: int = DEFAULT_MAX_CANDIDATES,
           epistemic_threshold_bits: float = EPISTEMIC_THRESHOLD_BITS) -> Dict[str, Any]:
    if model_key not in SUPPORTED_MODELS:
        raise UnknownModelError(f"'{model_key}' is not supported. Valid: {SUPPORTED_MODELS}")

    models = get_xai_models()
    meta = models[model_key]
    features = meta["features"]
    
    missing_fields = [f for f in features
                      if record.get(f) is None or (isinstance(record.get(f), float)
                                                   and pd.isna(record.get(f)))]
    present_count = len(features) - len(missing_fields)
    completeness_floor = float(meta.get("completeness_floor", 0.0))

    if present_count / len(features) < completeness_floor:
        missing_label = ", ".join(_field_label(field) for field in missing_fields[:3])
        return {
            "model": model_key,
            "doctor_explanation": [
                "Prediction: A reliable prediction cannot be made yet.",
                f"Estimated likelihood: Not available; only {present_count} of {len(features)} expected details are recorded.",
                "Why: The model should not be interpreted with this amount of missing information.",
                "Meaning: A model estimate can be provided once enough information is recorded; it is not a diagnosis.",
                f"Next useful detail: record {missing_label}.",
            ],
        }

    proba, classes = predict_proba_row(model_key, record)
    pred_class, pred_prob, confidence_probability = decision_from_proba(model_key, proba, classes)
    confidence = _confidence_label(confidence_probability)

    cf_results = []
    for field in features:
        if field not in missing_fields:
            result = simulate_known_field(model_key, record, field, k=k,
                                          max_candidates=max_candidates)
            if result is not None:
                cf_results.append(result)
    cf_results.sort(key=lambda result: result["max_confidence_shift"], reverse=True)

    if cf_results:
        hinge = cf_results[0]
        field_name = _field_label(hinge["field"])
        current_value = record.get(hinge["field"])
        if hinge["any_flip"]:
            alternative = next(item for item in hinge["alternatives"] if item["flips_prediction"])
            reason_line = (f"Why: The estimate is mainly influenced by {field_name}, recorded as "
                           f"'{current_value}'. A plausible alternative, "
                           f"'{alternative['alternative_value']}', would change the result to "
                           f"{alternative['resulting_prediction']}.")
        else:
            reason_line = (f"Why: The estimate is mainly influenced by {field_name}, recorded as "
                           f"'{current_value}'. Plausible alternatives change the confidence, "
                           f"but not the predicted result.")
    else:
        reason_line = "Why: No single recorded detail clearly determines this estimate; it reflects the combination of available information."

    if missing_fields:
        voi_results = []
        for field in missing_fields:
            result = simulate_missing_field(model_key, record, field, k=k,
                                            max_candidates=max_candidates)
            if result is not None:
                voi_results.append(result)
        voi_results.sort(key=lambda result: result["expected_reduction_bits"], reverse=True)
        if voi_results and voi_results[0]["expected_reduction_bits"] >= epistemic_threshold_bits:
            missing_line = (f"Next useful detail: {_field_label(voi_results[0]['field'])} is not recorded; "
                            f"adding it would most likely clarify this estimate.")
        else:
            missing_line = (f"Data note: {len(missing_fields)} expected detail(s) are not recorded, "
                            "but none is expected to materially resolve the uncertainty.")
    else:
        missing_line = "Data note: All expected details for this model are recorded."

    probability_line = (
        f"Estimated transfer likelihood: {pred_prob:.0%}. This is a {confidence}-confidence model estimate, not a clinical certainty."
        if model_key == "transfer" else
        f"Estimated likelihood: {pred_prob:.0%}. This is a {confidence}-confidence model estimate, not a clinical certainty."
    )

    # We use label map for pred_class mapping for the UI
    label_map = {str(k): str(v) for k, v in (meta.get("label_map") or {}).items()}
    display_pred_class = label_map.get(str(pred_class), str(pred_class))

    return {
        "model": model_key,
        "predicted_class": display_pred_class,
        "doctor_explanation": [
            _prediction_label(model_key, display_pred_class),
            probability_line,
            reason_line,
            _interpretation_line(model_key, display_pred_class),
            missing_line,
        ],
    }
