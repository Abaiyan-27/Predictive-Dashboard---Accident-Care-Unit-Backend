"""Each model receives the inputs its notebook trained on and returns only valid outputs.

The reference functions below are copied from the notebooks in
Group-01---Notebooks/Notebooks (text cleaning, rare-category consolidation,
ordinal encoding, severity feature engineering, decision rules). Real dataset
rows are scored both ways against the deployed artifacts; the backend must
reproduce the notebook's probabilities, labels and completeness exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from app.contracts.severity import SEVERITY_LABELS
from app.services import ml_service, patient_service

LOCAL = ml_service.LOCAL_MODEL_ROOT
SAMPLE_SIZE = 300
VALID_STATUSES = {"OK", "LOW_CONFIDENCE", "REFUSED", "UNAVAILABLE", "ERROR"}
WARD_COLS = ["Number of days in first ward", "Number of days in Second ward",
             "Number of days in Third ward"]
BUNDLE_LABELS = {
    "stay_duration": {"0-1 days", "2-3 days", "4+ days"},
    "transfer": {"Transfer", "No transfer"},
    "discharge": {"Complete recovery", "Partial recovery", "Further interventions"},
}


@pytest.fixture(autouse=True)
def local_model_root(monkeypatch):
    """Score the packaged artifacts regardless of what earlier API tests downloaded."""
    monkeypatch.setattr(ml_service, "MODEL_ROOT", LOCAL)


# ── Notebook reference implementations ─────────────────────────────────────

def notebook_clean(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "str":
            df[c] = (df[c].astype(str).str.strip().str.lower()
                     .replace({"nan": np.nan, "none": np.nan, "": np.nan}))
    return df


def notebook_total_days(df: pd.DataFrame) -> pd.Series:
    out = df.copy()
    for c in WARD_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        early = out[f"{c}_missing_reason"] == "early_discharge"
        out.loc[early, c] = out.loc[early, c].fillna(0)
    return out[WARD_COLS].sum(axis=1, min_count=1)


def notebook_bundle_proba(meta: dict, model, X: pd.DataFrame) -> np.ndarray:
    features = meta["features"]
    X = X[features].copy()
    for col, keep in meta.get("rare_maps", {}).items():
        X[col] = X[col].where(X[col].isna() | X[col].isin(set(keep)), "other")
    for col, order in [("Age", meta["age_order"]),
                       ("Time Taken To Reach Hospital", meta["time_order"])]:
        if col in features:
            X[col] = X[col].map({v: i for i, v in enumerate(order)})
    return model.predict_proba(X)


def to_region(site):
    if pd.isna(site):
        return np.nan
    if "no injury" in site or "no secondary" in site:
        return "none"
    if any(k in site for k in ["head", "forehead", "facial", "eye"]):
        return "head_face"
    if any(k in site for k in ["cervical", "thoracic", "lumbar", "sacrum"]):
        return "spine"
    if "chest" in site:
        return "chest"
    if "abdomen" in site:
        return "abdomen"
    if "pelvis" in site:
        return "pelvis"
    if any(k in site for k in ["shoulder", "clavicle", "humerus", "elbow", "forearm",
                               "hand", "carpal"]):
        return "upper_limb"
    if any(k in site for k in ["thigh", "femur", "knee", "leg", "foot", "tarsal",
                               "patella", "fibula", "tibia"]):
        return "lower_limb"
    return "other"


HIGH_RISK_REGIONS = {"head_face", "spine", "chest", "abdomen", "pelvis"}
NIGHT_WINDOWS = {"21:00 - 00:00", "00:00 - 03:00", "03:00 - 06:00"}
VULNERABLE_USERS = {"motor cyclist", "motorbike pillion rider", "bicycle",
                    "bicycle pillion", "pedestrian"}
HEAVY_COUNTERPART = {"heavy vehicle", "car or van", "three wheeler", "bus", "train"}


def isin_or_nan(series, values):
    return series.map(lambda v: np.nan if pd.isna(v) else (v in values))


def notebook_engineer(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["injury_region_1"] = out["Site of Injury No1"].map(to_region)
    out["injury_region_2"] = out["Site of injury No 2"].map(to_region)
    out["torso_head_injury"] = np.where(
        out["injury_region_1"].isna() & out["injury_region_2"].isna(), np.nan,
        (out["injury_region_1"].isin(HIGH_RISK_REGIONS) |
         out["injury_region_2"].isin(HIGH_RISK_REGIONS)).astype(float))
    out["multiple_injury_sites"] = np.where(
        out["injury_region_2"].isna(), np.nan,
        (~out["injury_region_2"].eq("none")).astype(float))
    out["night_collision"] = isin_or_nan(out["Time of Collision"], NIGHT_WINDOWS).astype(float)
    out["vulnerable_road_user"] = isin_or_nan(
        out["Mode Of Travel During Accident"], VULNERABLE_USERS).astype(float)
    out["heavy_counterpart"] = isin_or_nan(out["Collision With "], HEAVY_COUNTERPART).astype(float)
    out["ambulance_arrival"] = np.where(
        out["Mode of Transport to the Hospital"].isna(), np.nan,
        out["Mode of Transport to the Hospital"].eq("ambulance").astype(float))
    return out


def notebook_severity_proba(bundle: dict, raw: pd.DataFrame) -> np.ndarray:
    row = notebook_engineer(raw)
    for col, order in bundle["ordinal_orders"].items():
        row[col] = row[col].map({v: i for i, v in enumerate(order)})
    row = row[bundle["features"]]
    for c in bundle["cat_features"]:
        row[c] = row[c].astype(object)
    return bundle["model"].predict_proba(row)[:, 1]


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset():
    raw = pd.read_csv(LOCAL / "Preprocessed_Dataset.csv", low_memory=False)
    sample = raw.sample(SAMPLE_SIZE, random_state=7)
    clean = notebook_clean(sample)
    clean["total_days"] = notebook_total_days(clean)
    return sample, clean


def contract(name: str) -> dict:
    return json.loads((LOCAL / name / "model_contract.json").read_text())


def bundle_meta(name: str) -> tuple[dict, object]:
    c = contract(name)
    return joblib.load(LOCAL / name / c["metadata_artifact"]), joblib.load(LOCAL / name / c["artifact"])


def uglify(value):
    """Upper-case and pad text, as a form or a data entry might, to exercise normalisation."""
    return f"  {value.upper()} " if isinstance(value, str) else value


def backend_record(raw_row: pd.Series, total_days) -> dict:
    record = {key: uglify(value) for key, value in raw_row.items()}
    record["Collision With"] = record.get("Collision With ")
    record["total_days"] = total_days
    return record


# ── Notebook fidelity: bundle models ─────────────────────────────────────────

@pytest.mark.parametrize("name", ["stay_duration", "transfer", "discharge"])
def test_bundle_matches_notebook(name, dataset):
    raw, clean = dataset
    meta, model = bundle_meta(name)
    c = contract(name)
    label_map = c.get("label_map") or {}
    features = meta["features"]

    clean_renamed = clean.rename(columns={"Collision With ": "Collision With"})
    reference = notebook_bundle_proba(meta, model, clean_renamed)
    classes = [str(v) for v in model.classes_]
    completeness = clean_renamed[features].notna().sum(axis=1) / len(features)

    compared = 0
    for position, (index, raw_row) in enumerate(raw.iterrows()):
        result = ml_service.predict_bundle(
            name, backend_record(raw_row, clean.loc[index, "total_days"])
        )
        assert result.completeness == round(float(completeness.loc[index]), 3)
        required_missing = [
            f for f in ml_service.REQUIRED_INPUTS[name] if pd.isna(clean_renamed.loc[index, f])
        ]
        if required_missing or completeness.loc[index] < meta["completeness_floor"]:
            assert result.status == "REFUSED" and result.prediction is None
            continue

        expected = {label_map.get(cls, cls): reference[position][i] for i, cls in enumerate(classes)}
        assert result.probabilities.keys() == expected.keys()
        for label, probability in expected.items():
            assert result.probabilities[label] == pytest.approx(probability, abs=1e-4)

        if name == "transfer":
            p_yes = reference[position][classes.index("yes")]
            expected_label = "Transfer" if p_yes >= meta["chosen_threshold"] else "No transfer"
            assert result.probability == pytest.approx(p_yes, abs=1e-4)
            assert result.status == "OK"
        else:
            expected_label = label_map.get(classes[int(np.argmax(reference[position]))],
                                           classes[int(np.argmax(reference[position]))])
            expected_status = "OK" if reference[position].max() >= meta["low_conf_prob"] else "LOW_CONFIDENCE"
            assert result.status == expected_status
        assert result.prediction == expected_label
        compared += 1

    assert compared >= 50, f"only {compared} rows were scorable for {name}"


# ── Notebook fidelity: severity ────────────────────────────────────────────

def test_severity_matches_notebook(dataset):
    raw, clean = dataset
    bundle = joblib.load(LOCAL / "severity" / contract("severity")["artifact"])
    raw_inputs = bundle["raw_inputs"]
    reference = notebook_severity_proba(bundle, clean[raw_inputs])
    completeness = clean[raw_inputs].notna().sum(axis=1) / len(raw_inputs)

    for position, (index, raw_row) in enumerate(raw.iterrows()):
        record = {f: uglify(raw_row[f]) for f in raw_inputs}
        result = ml_service.predict_severity(record=record)
        assert result.prediction in SEVERITY_LABELS
        assert result.completeness == round(float(completeness.loc[index]), 3)
        if completeness.loc[index] < bundle["completeness_floor"]:
            assert result.prediction == "DEFER TO CLINICAL JUDGEMENT"
            assert result.status == "REFUSED"
            continue
        p = reference[position]
        expected = ("HIGH" if p >= bundle["high_cut"]
                    else "LOW" if p < bundle["low_cut"] else "BORDERLINE")
        assert result.probability == pytest.approx(p, abs=1e-4)
        assert result.prediction == expected


# ── Exact feature frame handed to each model ───────────────────────────────

class SpyModel:
    def __init__(self, model):
        self._model = model
        self.classes_ = model.classes_
        self.frames: list[pd.DataFrame] = []

    def predict_proba(self, frame):
        self.frames.append(frame.copy())
        return self._model.predict_proba(frame)


FULL_RECORD = {
    "Time of Collision": "21:00 - 00:00",
    "Age": "30-39",
    "Mode Of Travel During Accident": "Motor Cyclist",
    "Collision With ": "heavy vehicle",
    "Collision With": "heavy vehicle",
    "Vehicle Type": "motor cyclist",
    "Helmet Worn": "no",
    "Passenger Type": "driver",
    "Mode of Transport to the Hospital": "ambulance",
    "Time Taken To Reach Hospital": "30 minutes - 1 hour",
    "Site of Injury No1": "pelvis ramous",          # rare in training -> "other"
    "Site of injury No 2": None,                     # genuinely unrecorded
    "Other Injury": None,
    "Approximate Speed": "40 - 80 km/h",
    "Road Type": "bend",
    "Road Condition": "good",
    "Visiblity": "adequate",
    "Road Signals Exist": "no",
    "Collision Force From": "front",
    "Alcohol Consumption": "no",
    "Illicit Drugs": "no",
    "Type of injury No 1": "fracture",
    "Type of Injury No 2": None,
    "Transfer To Next Hospital": "no",
    "State of Transfer": "tranfer without any support",
    "Investigation Done": "x-ray",
    "Management Name 1": "fracture stabilization",
    "Management Name 2": "no other management",
    "total_days": 3,
}


@pytest.mark.parametrize("name", ["stay_duration", "transfer", "discharge"])
def test_bundle_receives_exact_feature_frame(name, monkeypatch):
    meta, model = bundle_meta(name)
    spy = SpyModel(model)
    monkeypatch.setattr(ml_service, "_load_bundle_model", lambda *_: (spy, None))

    result = ml_service.predict_bundle(name, dict(FULL_RECORD))
    assert result.status in {"OK", "LOW_CONFIDENCE"}
    frame = spy.frames[-1]

    assert list(frame.columns) == meta["features"]
    assert len(frame) == 1
    row = frame.iloc[0]
    for feature in meta["features"]:
        value = row[feature]
        if feature == "Age":
            assert value == meta["age_order"].index("30-39")
        elif feature == "Time Taken To Reach Hospital":
            assert value == meta["time_order"].index("30 minutes - 1 hour")
        elif feature == "total_days":
            assert value == 3
        elif FULL_RECORD[feature] is None:
            assert pd.isna(value), feature
        elif feature in meta.get("rare_maps", {}) and FULL_RECORD[feature] not in meta["rare_maps"][feature]:
            assert value == "other", feature
        else:
            assert value == FULL_RECORD[feature].lower(), feature
    for feature in meta["cat_features"]:
        assert frame[feature].dtype == object, feature


def test_severity_receives_exact_feature_frame(monkeypatch):
    bundle = joblib.load(LOCAL / "severity" / contract("severity")["artifact"])
    spy = SpyModel(bundle["model"])
    monkeypatch.setattr(ml_service, "_load_bundle_model", lambda *_: ({**bundle, "model": spy}, None))

    result = ml_service.predict_severity(record=dict(FULL_RECORD))
    assert result.prediction in SEVERITY_LABELS and result.status != "ERROR"
    row = spy.frames[-1].iloc[0]

    assert list(spy.frames[-1].columns) == bundle["features"]
    assert row["Age"] == 3
    assert row["Time Taken To Reach Hospital"] == 2
    assert row["Mode Of Travel During Accident"] == "motor cyclist"
    assert row["injury_region_1"] == "pelvis"
    assert pd.isna(row["Site of injury No 2"]) and pd.isna(row["injury_region_2"])
    assert pd.isna(row["multiple_injury_sites"])
    assert pd.isna(row["Other Injury"])
    assert row["torso_head_injury"] == 1.0
    assert row["night_collision"] == 1.0
    assert row["vulnerable_road_user"] == 1.0
    assert row["heavy_counterpart"] == 1.0
    assert row["ambulance_arrival"] == 1.0
    assert result.completeness == round(10 / 12, 3)


# ── Database rows -> model record ────────────────────────────────────────────

PATIENT = {"patient_id": "p1", "age": "34", "gender": "Male"}
INTAKE_ACCIDENT = {
    "accident_id": "a1",
    "incident_at": "2026-09-01T22:15:00+00:00",
    "mode_of_travel_during_accident": "motor cyclist",
    "collision_with": "heavy vehicle",
    "vehicle_type": "motor cyclist",
    "helmet_worn": "no",
    "passenger_type": "driver",
    "mode_of_transport": "ambulance",
    "time_to_hospital": "30 minutes - 1 hour",
    "speed": "40 - 80 km/h",
    "road_type": "bend",
    "road_condition": "good",
    "visibility": "adequate",
    "road_signals_exist": "no",
    "collision_force_from": "front",
    "alcohol_consumption": "no",
    "illicit_drugs": "no",
    "other_injury": None,
    "Injury_Description": [{"site_of_injury": "head injury", "type_of_injury": "laceration"}],
}
TREATMENT = {
    "investigation_done": "x-ray",
    "management_name_1": "wound suturing",
    "management_name_2": "no other management",
    "number_of_days_stay": 2,
    "created_at": "2026-09-02T08:00:00+00:00",
}


def test_record_contains_every_model_input():
    record = patient_service.build_model_record(patient=PATIENT, accident=INTAKE_ACCIDENT)
    bundle = joblib.load(LOCAL / "severity" / contract("severity")["artifact"])
    assert set(bundle["raw_inputs"]) <= record.keys()
    for name in ("stay_duration", "transfer", "discharge"):
        meta, _ = bundle_meta(name)
        assert set(meta["features"]) <= record.keys(), name


def test_intake_record_has_no_invented_values():
    record = patient_service.build_model_record(patient=PATIENT, accident=INTAKE_ACCIDENT)
    assert record["Age"] == "30-39"
    assert record["Time of Collision"] == "21:00 - 00:00"
    assert record["Site of Injury No1"] == "head injury"
    for field in ("Site of injury No 2", "Type of Injury No 2", "Other Injury",
                  "Transfer To Next Hospital", "State of Transfer", "Investigation Done",
                  "Management Name 1", "Management Name 2", "total_days"):
        assert record[field] is None, field


def test_severity_completeness_counts_only_entered_fields():
    accident = {"Injury_Description": [{"site_of_injury": "head injury", "type_of_injury": None}]}
    record = patient_service.build_model_record(patient=PATIENT, accident=accident)
    result = ml_service.predict_severity(record=record)
    assert result.completeness == round(2 / 12, 3)  # Age + primary site, nothing else


def test_age_only_defers():
    record = patient_service.build_model_record(patient=PATIENT, accident={})
    result = ml_service.predict_severity(record=record)
    assert result.prediction == "DEFER TO CLINICAL JUDGEMENT"
    assert result.status == "REFUSED"


def test_treatment_without_transfer_uses_dataset_encoding():
    accident = {**INTAKE_ACCIDENT, "Treatment": [TREATMENT, {"number_of_days_stay": None}]}
    record = patient_service.build_model_record(patient=PATIENT, accident=accident, transfers=[])
    assert record["Transfer To Next Hospital"] == "no"
    assert record["State of Transfer"] == "tranfer without any support"
    assert record["total_days"] == 2  # a treatment row without days adds nothing


def test_open_transfer_leaves_transfer_inputs_unknown():
    accident = {**INTAKE_ACCIDENT, "Treatment": [TREATMENT]}
    record = patient_service.build_model_record(
        patient=PATIENT, accident=accident, transfers=[{"status": "pending_authorization"}],
    )
    assert record["Transfer To Next Hospital"] is None
    assert record["State of Transfer"] is None


def test_approved_transfer_is_recorded():
    record = patient_service.build_model_record(
        patient=PATIENT, accident=INTAKE_ACCIDENT,
        transfers=[{"status": "rejected", "state_of_transfer": "x"},
                   {"status": "approved", "state_of_transfer": "spinal board used during transfer"}],
    )
    assert record["Transfer To Next Hospital"] == "yes"
    assert record["State of Transfer"] == "spinal board used during transfer"


# ── Post-treatment gate ───────────────────────────────────────────────────

def test_intake_does_not_run_post_treatment_models(monkeypatch):
    calls = []
    original = ml_service._load_bundle_model
    monkeypatch.setattr(
        ml_service, "_load_bundle_model",
        lambda name, c: calls.append(name) or original(name, c),
    )
    accident = patient_service.add_patient_predictions(
        {**INTAKE_ACCIDENT, "Patient": PATIENT}
    )
    for key in ("stay_duration_prediction", "transfer_prediction"):
        prediction = accident[key]
        assert prediction["status"] == "REFUSED"
        assert prediction["prediction"] is None
        assert "Waiting for post-treatment inputs" in prediction["message"]
    assert accident["severity_prediction"]["prediction"] in SEVERITY_LABELS
    assert calls == ["severity"]


def test_post_treatment_models_run_once_treatment_is_recorded():
    accident = {**INTAKE_ACCIDENT, "Treatment": [TREATMENT]}
    record = patient_service.build_model_record(patient=PATIENT, accident=accident, transfers=[])
    for name, fn in (("stay_duration", ml_service.predict_stay_duration),
                     ("transfer", ml_service.predict_transfer_probability),
                     ("discharge", ml_service.predict_discharge_outcome)):
        result = fn(record)
        assert result.status in {"OK", "LOW_CONFIDENCE"}, (name, result.message)
        assert result.prediction in BUNDLE_LABELS[name]


# ── Output validity across real records ────────────────────────────────────

def test_every_output_is_valid(dataset):
    raw, clean = dataset
    for index, raw_row in raw.iterrows():
        record = backend_record(raw_row, clean.loc[index, "total_days"])
        severity = ml_service.predict_severity(record=record)
        assert severity.prediction in SEVERITY_LABELS
        assert severity.status in VALID_STATUSES
        for name in BUNDLE_LABELS:
            result = ml_service.predict_bundle(name, record)
            assert result.status in VALID_STATUSES
            if result.prediction is None:
                assert result.status == "REFUSED"
                continue
            assert result.prediction in BUNDLE_LABELS[name]
            assert set(result.probabilities) == BUNDLE_LABELS[name]
            assert sum(result.probabilities.values()) == pytest.approx(1, abs=1e-3)
            assert 0 <= result.probability <= 1


@pytest.mark.parametrize("status, patch_target", [
    ("UNAVAILABLE", "_load_bundle_model"),
    ("ERROR", "_engineer_severity_row"),
])
def test_severity_failures_still_return_a_contract_label(status, patch_target, monkeypatch):
    if patch_target == "_load_bundle_model":
        monkeypatch.setattr(ml_service, patch_target, lambda *_: (None, "missing"))
    else:
        def boom(_):
            raise RuntimeError("broken")
        monkeypatch.setattr(ml_service, patch_target, boom)
    result = ml_service.predict_severity(record=dict(FULL_RECORD))
    assert result.status == status
    assert result.prediction == "DEFER TO CLINICAL JUDGEMENT"


# Explainability (SHAP) is tested in tests/unit/test_shap_explainer.py.
