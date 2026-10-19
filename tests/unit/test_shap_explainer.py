"""The SHAP explainer reproduces Notebooks/Doctor_SHAP_Explainability.ipynb exactly.

The notebook's own explainer code (section 1) is executed against the deployed
model files and compared with app/services/shap_explainer.py: same background,
same SHAP values, same probabilities. The only intended difference is that the
backend explains the label the deployed rule shows (Transfer uses the tuned
threshold), where the notebook explains the argmax.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from app.services import ml_service, shap_explainer

pytest.importorskip("shap")

BACKEND = Path(__file__).resolve().parents[2]
NOTEBOOK = BACKEND / "Notebooks" / "Doctor_SHAP_Explainability.ipynb"
LOCAL = ml_service.LOCAL_MODEL_ROOT
NOTEBOOK_FILES = {   # notebook file name -> deployed bundle file
    "discharge_outcome_model.joblib": "discharge/discharge_outcome_model_notebook.joblib",
    "discharge_outcome_metadata.joblib": "discharge/discharge_outcome_metadata_notebook.joblib",
    "transfer_model.joblib": "transfer/transfer_model_notebook.joblib",
    "transfer_metadata.joblib": "transfer/transfer_metadata_notebook.joblib",
    "stay_duration_model.joblib": "stay_duration/stay_duration_model_notebook.joblib",
    "stay_duration_metadata.joblib": "stay_duration/stay_duration_metadata_notebook.joblib",
}


@pytest.fixture(autouse=True)
def packaged_models(monkeypatch):
    monkeypatch.setattr(ml_service, "MODEL_ROOT", LOCAL)


@pytest.fixture(scope="module")
def notebook(tmp_path_factory):
    """The notebook's section 1, run on the deployed model files."""
    model_dir = tmp_path_factory.mktemp("notebook_artifacts")
    for name, deployed in NOTEBOOK_FILES.items():
        shutil.copy(LOCAL / deployed, model_dir / name)
    code = "".join(json.loads(NOTEBOOK.read_text())["cells"][2]["source"])
    namespace: dict = {}
    exec(code, namespace)  # noqa: S102 - the repository's own notebook
    namespace["MODEL_DIR"] = model_dir
    namespace["DATA_PATH"] = LOCAL / "Preprocessed_Dataset.csv"
    patients = namespace["load_patient_table"](namespace["DATA_PATH"])
    explainers = {key: namespace["DoctorExplainer"](key, model_dir, patients) for key in namespace["MODEL_FILES"]}
    return {"explainers": explainers, "patients": patients, "names": namespace["DISPLAY_NAMES"]}


@pytest.fixture(scope="module")
def ours():
    return shap_explainer.get_explainers()


def rows(notebook, step):
    patients = notebook["patients"]
    return [patients.iloc[i].to_dict() for i in range(0, len(patients), step)]


@pytest.mark.parametrize("key", shap_explainer.SUPPORTED_MODELS)
def test_background_is_the_notebooks(notebook, ours, key):
    theirs = notebook["explainers"][shap_explainer.NOTEBOOK_KEYS[key]].background
    assert ours[key].background.shape == (300, theirs.shape[1])
    assert np.array_equal(ours[key].background, theirs)


@pytest.mark.parametrize("key", shap_explainer.SUPPORTED_MODELS)
def test_shap_values_and_probabilities_are_the_notebooks(notebook, ours, key):
    mine, theirs = ours[key], notebook["explainers"][shap_explainer.NOTEBOOK_KEYS[key]]
    raw_name = {display: raw for raw, display in notebook["names"].items()}
    compared = 0
    for record in rows(notebook, 23):
        result = mine.explain(record)
        if not result["available"]:
            continue
        expected = theirs.explain(record)
        if mine._display(expected["prediction"]) != result["prediction"]:
            continue          # threshold band: covered by the transfer test below
        def by_factor(items, rename):
            return {rename(item["factor"]): item["shap"] for item in items}
        got = by_factor(result["points_towards"] + result["points_away"], lambda f: f)
        want = by_factor(expected["points_towards"] + expected["points_away"],
                         lambda f: shap_explainer.DISPLAY_NAMES.get(raw_name.get(f, f), f))
        assert got == want
        assert result["probabilities"] == {mine._display(k): v for k, v in expected["probabilities"].items()}
        compared += 1
    assert compared > 40


@pytest.mark.parametrize("key", shap_explainer.SUPPORTED_MODELS)
def test_explanations_are_exact(ours, key):
    """The notebook's check: base value + SHAP values = the model's log-odds."""
    explainer = ours[key]
    x = explainer.background[:100]
    margins = explainer.clf.decision_function(x)
    base = np.atleast_1d(explainer.explainer.expected_value)
    values = explainer.explainer.shap_values(x)
    if margins.ndim == 1:
        array = np.asarray(values[0] if isinstance(values, list) else values)
        error = np.abs(base[0] + array.sum(1) - margins).max()
    else:
        array = np.stack(values, -1) if isinstance(values, list) else np.asarray(values)
        error = np.abs(base + array.sum(1) - margins).max()
    assert error < 1e-9


@pytest.mark.parametrize("key", shap_explainer.SUPPORTED_MODELS)
def test_the_explained_label_is_the_deployed_prediction(notebook, ours, key):
    for record in rows(notebook, 97):
        result = ours[key].explain(record)
        prediction = ml_service.predict_bundle(key, record)
        assert result["available"] is (prediction.prediction is not None)
        if result["available"]:
            assert result["prediction"] == prediction.prediction
            assert result["doctor_text"].startswith(f"{shap_explainer.MODEL_TITLES[key]}: ")


def test_transfer_explains_the_thresholded_label_not_the_argmax(notebook, ours):
    """Between 50% and the 62% threshold the screen says "No transfer"; so must the explanation."""
    mine, theirs = ours["transfer"], notebook["explainers"]["transfer"]
    threshold = mine.settings["decision_threshold"]
    for record in rows(notebook, 3):
        result = mine.explain(record)
        if result["available"] and 0.5 <= result["probabilities"]["Transfer"] < threshold:
            break
    else:
        pytest.fail("no record in the threshold band")
    expected = theirs.explain(record)
    assert expected["prediction"] == "yes"                      # the notebook's argmax
    assert result["prediction"] == "No transfer"                # what the doctor sees
    assert result["headline"].startswith("Transfer unlikely") and "62%" in result["headline"]
    # Binary model: attributions for "no" are exactly those for "yes", negated.
    ours_values = {i["factor"]: i["shap"] for i in result["points_towards"] + result["points_away"]}
    theirs_values = {shap_explainer.DISPLAY_NAMES.get(i["factor"], i["factor"]): i["shap"]
                     for i in expected["points_towards"] + expected["points_away"]}
    shared = set(ours_values) & set(theirs_values)
    assert shared and all(ours_values[f] == pytest.approx(-theirs_values[f]) for f in shared)


def test_no_explanation_until_the_model_can_run(ours):
    result = ours["transfer"].explain({"Age": "30-39"})
    assert result["available"] is False and result["message"]
    assert "points_towards" not in result


def test_unknown_models_are_refused_and_the_notebook_key_is_accepted(ours):
    with pytest.raises(shap_explainer.UnknownModelError):
        shap_explainer.resolve_model("severity")
    assert shap_explainer.resolve_model("stay") == "stay_duration"


def test_responses_name_no_server_paths(notebook, ours):
    record = rows(notebook, 500)[1]
    text = json.dumps(shap_explainer.explain_all(record))
    assert "/home/" not in text and "/tmp/" not in text and ".joblib" not in text


def test_explainers_follow_the_active_model_root(tmp_path, monkeypatch):
    for name in ("stay_duration", "transfer", "discharge"):
        shutil.copytree(LOCAL / name, tmp_path / name)
    shutil.copy(LOCAL / "Preprocessed_Dataset.csv", tmp_path / "Preprocessed_Dataset.csv")
    shap_explainer.get_explainers()                 # built on the packaged root
    monkeypatch.setattr(ml_service, "MODEL_ROOT", tmp_path)
    ml_service._bundle_cache.clear()
    shap_explainer.get_explainers()
    assert shap_explainer._cache_root == tmp_path
    assert shap_explainer.training_data_path() == tmp_path / "Preprocessed_Dataset.csv"
    for name in ("stay_duration", "transfer", "discharge"):
        artifact = json.loads((LOCAL / name / "model_contract.json").read_text())["artifact"]
        assert str(tmp_path / name / artifact) in ml_service._bundle_cache
    (tmp_path / "Preprocessed_Dataset.csv").unlink()
    assert shap_explainer.training_data_path() == LOCAL / "Preprocessed_Dataset.csv"
