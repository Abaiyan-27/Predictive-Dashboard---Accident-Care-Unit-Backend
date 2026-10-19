"""SHAP explanations for the doctor models (Stay Duration, Transfer, Discharge).

Ported from Notebooks/Doctor_SHAP_Explainability.ipynb. Each doctor model is a
scikit-learn pipeline ("pre" + LogisticRegression), so shap.LinearExplainer
gives exact attributions: the base value plus the SHAP values equals the
model's own log-odds. As in the notebook, the background is 300 patients drawn
from Preprocessed_Dataset.csv with random_state 42, prepared the way the model
notebooks prepared their training data.

Two things are taken from the deployed system rather than recomputed, so an
explanation always describes the prediction the doctor sees:

* the model, metadata and model root come from ml_service (the same cache,
  including a Hugging Face bundle when one is configured);
* the patient row is prepared by ml_service._prepare_bundle_row, the exact
  inference path, and the explained class is the one the deployed rule selected
  (for Transfer that is the notebook's tuned threshold, not the argmax).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.services import ml_service

logger = logging.getLogger(__name__)

BACKGROUND_SIZE = 300
RANDOM_STATE = 42
TOP_K = 5

# Backend model key -> notebook model key.
NOTEBOOK_KEYS = {"stay_duration": "stay", "transfer": "transfer", "discharge": "discharge"}
ALIASES = {"stay": "stay_duration"}
SUPPORTED_MODELS = tuple(NOTEBOOK_KEYS)

MODEL_TITLES = {
    "discharge": "Discharge outcome",
    "transfer": "Transfer to another hospital",
    "stay_duration": "Expected hospital stay",
}
CLASS_TEXT = {("transfer", "yes"): "Transfer likely", ("transfer", "no"): "Transfer unlikely"}
OUTCOME_TEXT = {("transfer", "yes"): "transfer", ("transfer", "no"): "no transfer"}

DISPLAY_NAMES = {
    "Age": "Age",
    "Gender": "Gender",
    "Mode of Transport to the Hospital": "How they reached hospital",
    "Time Taken To Reach Hospital": "Time taken to reach hospital",
    "Vehicle Type": "Vehicle type",
    "Helmet Worn": "Helmet worn",
    "Collision With": "What they collided with",
    "Collision Force From": "Direction of impact",
    "Approximate Speed": "Approximate speed",
    "Road Type": "Road type",
    "Road Condition": "Road condition",
    "Alcohol Consumption": "Alcohol consumption",
    "Site of Injury No1": "Main injury site",
    "Type of injury No 1": "Main injury type",
    "Site of injury No 2": "Second injury site",
    "Type of Injury No 2": "Second injury type",
    "Investigation Done": "Investigation done",
    "Management Name 1": "First treatment",
    "Management Name 2": "Second treatment",
    "Transfer To Next Hospital": "Transferred to next hospital",
    "State of Transfer": "State on transfer",
    "total_days": "Days in hospital so far",
    # Model features the notebook had no display name for (shown as raw column names there).
    "Visiblity": "Visibility",
    "Road Signals Exist": "Road signals",
    "Illicit Drugs": "Illicit drugs",
}

STRONG, MODERATE, SLIGHT = 0.5, 0.2, 0.05     # effect size thresholds (log-odds)
LOW_CONFIDENCE = 0.5
DISCLAIMER = "Decision support only. The final clinical decision rests with the doctor."
METHOD = "SHAP (LinearExplainer): exact attributions for this logistic-regression model"
WARD_COLUMNS = ["Number of days in first ward", "Number of days in Second ward", "Number of days in Third ward"]


class ExplainError(Exception):
    """Raised for problems that should surface as a clean API error."""


class UnknownModelError(ExplainError):
    """The requested model has no SHAP explainer."""


class ModelLoadError(ExplainError):
    """A model, its metadata or the SHAP background could not be loaded."""


# ── Helpers (as in the notebook) ──────────────────────────────────────────

def _normalise_text(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == object or pd.api.types.is_string_dtype(out[column]):
            out[column] = (out[column].astype(str).str.strip().str.lower()
                           .replace({"nan": np.nan, "none": np.nan, "": np.nan}))
    return out


def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _strength(value: float) -> str:
    size = abs(value)
    return "strong" if size >= STRONG else "moderate" if size >= MODERATE else "slight"


def load_patient_table(path: Path) -> pd.DataFrame:
    """Training data prepared the same way as in the model notebooks."""
    df = pd.read_csv(path, low_memory=False).rename(columns={"Collision With ": "Collision With"})
    df = _normalise_text(df)
    for column in WARD_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        reason = f"{column}_missing_reason"
        if reason in df.columns:
            early = df[reason] == "early_discharge"
            df.loc[early, column] = df.loc[early, column].fillna(0)
    df["total_days"] = df[WARD_COLUMNS].sum(axis=1, min_count=1)
    return df


def training_data_path() -> Path:
    """Prefer the dataset shipped with the active model bundle, else the packaged copy."""
    active = ml_service.get_model_root() / "Preprocessed_Dataset.csv"
    return active if active.exists() else ml_service.LOCAL_MODEL_ROOT / "Preprocessed_Dataset.csv"


def doctor_text(result: dict[str, Any]) -> str:
    lines = [f"{result['title']}: {result['headline']}"]
    if result["note"]:
        lines.append(f"  ! {result['note']}")
    if result["points_towards"]:
        lines.append(f"  Points towards {result['outcome']}:")
        lines += [f"    + {x['factor']}: {x['value']} ({x['effect']})" for x in result["points_towards"]]
    if result["points_away"]:
        lines.append(f"  Points away from {result['outcome']}:")
        lines += [f"    - {x['factor']}: {x['value']} ({x['effect']})" for x in result["points_away"]]
    if result["not_recorded"]:
        lines.append("  Not yet recorded: " + ", ".join(result["not_recorded"]))
    lines.append("  " + result["disclaimer"])
    return "\n".join(lines)


# ── One explainer per doctor model ────────────────────────────────────────

class DoctorExplainer:
    """SHAP explainer for one doctor model, built on the deployed artifact."""

    def __init__(self, model_key: str, patients: Optional[pd.DataFrame]):
        import shap
        from sklearn.linear_model import LogisticRegression

        self.key = model_key
        contract, error = ml_service._read_contract(model_key)
        if error or contract is None:
            raise ModelLoadError(f"Missing contract for {model_key}")
        settings, error = ml_service._bundle_settings(model_key, contract)
        if error or settings is None:
            raise ModelLoadError(f"Missing metadata for {model_key}")
        model, error = ml_service._load_bundle_model(model_key, contract)
        if error or model is None:
            raise ModelLoadError(f"Missing model for {model_key}")

        self.settings = settings
        self.features = list(settings["features"])
        steps = getattr(model, "named_steps", {})
        if "pre" not in steps or not isinstance(steps.get("clf"), LogisticRegression):
            raise ModelLoadError(f"{model_key} is not a 'pre' + LogisticRegression pipeline")
        self.model, self.pre, self.clf = model, steps["pre"], steps["clf"]
        self.classes = [str(c) for c in self.clf.classes_]
        self.label_map = {str(k): str(v) for k, v in (settings.get("label_map") or {}).items()}

        self.background = self._background(patients)
        self.explainer = shap.LinearExplainer(
            self.clf, shap.maskers.Independent(self.background, max_samples=len(self.background)))

        by_length = sorted(self.features, key=len, reverse=True)
        self.origin = []
        for name in self.pre.get_feature_names_out():
            base = name.split("__", 1)[-1]
            self.origin.append(next((f for f in by_length if base == f or base.startswith(f + "_")), base))

    # The notebook's preparation of training rows (used for the background).
    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = _normalise_text(frame.reindex(columns=self.features))
        for column, keep in (self.settings.get("rare_maps") or {}).items():
            if column in out.columns:
                out[column] = out[column].where(out[column].isna() | out[column].isin(set(keep)), "other")
        for column, mapping in (self.settings.get("ordinal_mappings") or {}).items():
            if column in out.columns:
                out[column] = out[column].map(mapping)
        return out

    def _background(self, patients: Optional[pd.DataFrame]) -> np.ndarray:
        if patients is None:
            raise ModelLoadError("The SHAP background dataset is not available")
        missing = [f for f in self.features if f not in patients.columns]
        if missing:
            raise ModelLoadError(f"{self.key} expects columns not in the dataset: {missing}")
        sample = patients[self.features].sample(min(BACKGROUND_SIZE, len(patients)), random_state=RANDOM_STATE)
        return _dense(self.pre.transform(self.prepare(sample)))

    def _shap_for_class(self, values, index: int) -> np.ndarray:
        if isinstance(values, list):
            return np.asarray(values[index])[0]
        array = np.asarray(values)
        if array.ndim == 3:
            return array[0, :, index]
        return array[0] if index == 1 or len(self.classes) > 2 else -array[0]

    def _display(self, raw_class: str) -> str:
        return self.label_map.get(raw_class, raw_class)

    def explain(self, record: dict[str, Any], top_k: int = TOP_K) -> dict[str, Any]:
        title = MODEL_TITLES[self.key]
        prediction = ml_service.predict_bundle(self.key, record)
        if prediction.prediction is None:
            # The model itself refused (e.g. treatment not yet recorded): nothing to explain.
            return {"model": self.key, "title": title, "available": False,
                    "message": prediction.message or "No prediction is available for this patient yet.",
                    "disclaimer": DISCLAIMER, "method": METHOD}

        row, _, row_error = ml_service._prepare_bundle_row(record, self.settings)
        if row is None:
            return {"model": self.key, "title": title, "available": False,
                    "message": row_error or "No prediction is available for this patient yet.",
                    "disclaimer": DISCLAIMER, "method": METHOD}
        x = _dense(self.pre.transform(row))
        proba = self.clf.predict_proba(x)[0]

        # Explain the class the deployed rule selected (the label on the doctor's screen).
        displayed = [self._display(c) for c in self.classes]
        index = displayed.index(prediction.prediction)
        raw_class = self.classes[index]
        values = self._shap_for_class(self.explainer.shap_values(x), index)
        per_feature = pd.Series(values, index=self.origin).groupby(level=0).sum()
        ranked = per_feature.reindex(per_feature.abs().sort_values(ascending=False).index)

        shown_row = _normalise_text(pd.DataFrame([{f: ml_service.normalise_record(record).get(f) for f in self.features}])).iloc[0]
        recorded = [f for f in ranked.index if pd.notna(shown_row[f])]
        shown = [(f, float(ranked[f])) for f in recorded if abs(ranked[f]) >= SLIGHT][:top_k]

        def item(feature: str, value: float) -> dict[str, Any]:
            return {"factor": DISPLAY_NAMES.get(feature, feature), "value": str(shown_row[feature]),
                    "effect": _strength(value), "shap": round(value, 4)}

        threshold = self.settings.get("decision_threshold")
        positive = self.settings.get("positive_class")
        if threshold is not None and positive is not None:
            positive_probability = float(proba[self.classes.index(str(positive))])
            headline = (f"{CLASS_TEXT.get((self.key, raw_class), prediction.prediction)} "
                        f"({positive_probability:.0%} probability of {self._display(str(positive)).lower()}; "
                        f"predicted at {float(threshold):.0%} or more)")
            low_confidence = False
        else:
            headline = f"{CLASS_TEXT.get((self.key, raw_class), prediction.prediction)} ({float(proba[index]):.0%} probability)"
            low_confidence = float(proba[index]) < LOW_CONFIDENCE

        note = None
        if not recorded:
            note = "No details recorded yet, so this reflects a typical patient."
        elif low_confidence:
            note = "Low confidence: the model is unsure, so treat this as a weak signal."

        result = {
            "model": self.key,
            "title": title,
            "available": True,
            "prediction": prediction.prediction,
            "headline": headline,
            "outcome": OUTCOME_TEXT.get((self.key, raw_class), prediction.prediction.lower()),
            "probabilities": {self._display(c): round(float(p), 4) for c, p in zip(self.classes, proba)},
            "points_towards": [item(f, v) for f, v in shown if v > 0],
            "points_away": [item(f, v) for f, v in shown if v < 0],
            "not_recorded": [DISPLAY_NAMES.get(f, f) for f in ranked.index if pd.isna(shown_row[f])][:4],
            "note": note,
            "disclaimer": DISCLAIMER,
            "method": METHOD,
        }
        result["doctor_text"] = doctor_text(result)
        return result

    def global_importance(self) -> dict[str, float]:
        """Average |SHAP| per feature over the background patients."""
        values = self.explainer.shap_values(self.background)
        array = np.abs(np.stack(values, -1) if isinstance(values, list) else np.asarray(values))
        array = array.mean(-1) if array.ndim == 3 else array
        importance = (pd.Series(array.mean(0), index=self.origin).groupby(level=0).sum()
                      .sort_values(ascending=False))
        return {DISPLAY_NAMES.get(k, k): round(float(v), 4) for k, v in importance.items()}


# ── Cache: built once per model root (rebuilt if inference switches roots) ─

_lock = threading.Lock()
_explainers: dict[str, DoctorExplainer] = {}
_cache_root: Optional[Path] = None


def get_explainers() -> dict[str, DoctorExplainer]:
    global _cache_root
    with _lock:
        root = ml_service.get_model_root()
        if root != _cache_root:
            _explainers.clear()
            _cache_root = root
        if len(_explainers) < len(SUPPORTED_MODELS):
            path = training_data_path()
            try:
                patients = load_patient_table(path)
            except (OSError, ValueError) as exc:
                raise ModelLoadError("The SHAP background dataset could not be read") from exc
            for key in SUPPORTED_MODELS:
                if key not in _explainers:
                    _explainers[key] = DoctorExplainer(key, patients)
        return dict(_explainers)


def resolve_model(model_key: str) -> str:
    key = ALIASES.get(model_key, model_key)
    if key not in SUPPORTED_MODELS:
        raise UnknownModelError(f"Unknown model '{model_key}'. Valid: {list(SUPPORTED_MODELS)}")
    return key


def explain(model_key: str, record: dict[str, Any]) -> dict[str, Any]:
    key = resolve_model(model_key)
    return get_explainers()[key].explain(record)


def explain_all(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {key: explainer.explain(record) for key, explainer in get_explainers().items()}
