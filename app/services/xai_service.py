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


def simulate_known_field(model_key: str, record: Dict[str, Any], field: str, k: int = DEFAULT_K_NEIGHBORS,
                         max_candidates: int = DEFAULT_MAX_CANDIDATES):
    current_proba, classes = predict_proba_row(model_key, record)
    pred_idx = int(np.argmax(current_proba))
    pred_class, pred_prob = str(classes[pred_idx]), float(current_proba[pred_idx])

    candidates = plausible_values(model_key, record, field, k=k, max_candidates=max_candidates,
                                  exclude_current=True)
    if not candidates:
        return None

    results = []
    for value, weight in candidates:
        modified = dict(record)
        modified[field] = value
        proba, _ = predict_proba_row(model_key, modified)
        new_idx = int(np.argmax(proba))
        confidence_drop = pred_prob - float(proba[pred_idx])
        results.append({"alternative_value": value, "resulting_prediction": str(classes[new_idx]),
                        "confidence_drop": round(confidence_drop, 4),
                        "confidence_shift_abs": round(abs(confidence_drop), 4),
                        "flips_prediction": classes[new_idx] != pred_class})

    most_sensitive = max(results, key=lambda r: r["confidence_shift_abs"])

    return {"field": field, "current_prediction": pred_class,
            "current_probability": round(pred_prob, 4), "alternatives": results,
            "max_confidence_shift": most_sensitive["confidence_shift_abs"],
            "most_sensitive_alternative": most_sensitive["alternative_value"],
            "any_flip": any(r["flips_prediction"] for r in results)}


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
    known_fields = [f for f in features if f not in missing_fields]

    proba, classes = predict_proba_row(model_key, record)
    pred_idx = int(np.argmax(proba))
    
    # Check if there's a label mapping
    label_map = {str(k): str(v) for k, v in (meta.get("label_map") or {}).items()}
    display_classes = [label_map.get(str(c), str(c)) for c in classes]
    
    pred_class, pred_prob = display_classes[pred_idx], float(proba[pred_idx])
    current_entropy = entropy_bits(proba)
    max_entropy = float(np.log2(len(classes)))

    result = {"model": model_key, "predicted_class": pred_class,
              "predicted_probability": round(pred_prob, 4),
              "entropy_bits": round(current_entropy, 4), "max_entropy_bits": round(max_entropy, 4),
              "n_missing_fields": len(missing_fields), "n_known_fields": len(known_fields)}

    if missing_fields:
        voi_results = []
        for f in missing_fields:
            r = simulate_missing_field(model_key, record, f, k=k, max_candidates=max_candidates)
            if r is not None:
                voi_results.append(r)
        voi_results.sort(key=lambda r: r["expected_reduction_bits"], reverse=True)
        result["value_of_information_ranking"] = voi_results

        if voi_results and voi_results[0]["expected_reduction_bits"] >= epistemic_threshold_bits:
            result["uncertainty_type"] = "epistemic"
            result["recommended_field"] = voi_results[0]["field"]
            result["message"] = (f"Uncertain ({pred_class}, {pred_prob:.0%}). Getting "
                                 f"'{voi_results[0]['field']}' would likely help most "
                                 f"(expected reduction: {voi_results[0]['expected_reduction_bits']:.2f} bits).")
        else:
            result["uncertainty_type"] = "aleatoric"
            result["message"] = (f"Uncertain ({pred_class}, {pred_prob:.0%}). No missing field is "
                                 f"expected to meaningfully resolve this - the case appears "
                                 f"genuinely borderline.")
    else:
        result["uncertainty_type"] = "not_applicable_fully_specified"
        result["value_of_information_ranking"] = []

    if known_fields:
        cf_results = []
        for f in known_fields:
            r = simulate_known_field(model_key, record, f, k=k, max_candidates=max_candidates)
            if r is not None:
                cf_results.append(r)
        cf_results.sort(key=lambda r: r["max_confidence_shift"], reverse=True)
        result["counterfactual_ranking"] = cf_results
        if cf_results:
            top = cf_results[0]
            result["hinge_field"] = top["field"]
            flip_note = ""
            if top["any_flip"]:
                flipped_to = next(a["resulting_prediction"] for a in top["alternatives"]
                                  if a["flips_prediction"])
                
                mapped_flipped_to = label_map.get(str(flipped_to), str(flipped_to))
                flip_note = f" A plausible alternative value would flip it to '{mapped_flipped_to}'."
            result["hinge_message"] = (f"This prediction is most sensitive to '{top['field']}' - "
                                       f"an alternative plausible value would shift confidence by "
                                       f"up to {top['max_confidence_shift']:.0%}.{flip_note}")
    else:
        result["counterfactual_ranking"] = []

    return result
