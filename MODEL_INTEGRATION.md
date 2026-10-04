# Plugging notebook models into the dashboard

FastAPI cannot serve an `.ipynb` file directly. Run the notebook, export its
final fitted model, and install that binary artifact into the matching model
bundle. Preprocessing and feature order are declared by each bundle's
`model_contract.json`.

## One-time application setup

1. Install the updated Python requirements.

   ```bash
   cd Predictive-Dashboard---Accident-Care-Unit-Backend
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Apply every SQL file in `supabase/migrations/` in filename order (Supabase
   CLI `supabase db push`, or paste each into the SQL editor). They are
   idempotent and abort with a list of offending rows rather than changing data
   that would violate a new constraint. They create the base tables, the model
   input columns, ordered injuries, the hospital network, and the append-only
   `Model_Prediction` / `Prediction_Decision` history tables.

   After running it, verify the live database at
   `http://localhost:8000/health/database-schema`. The top-level `ready` value
   must be `true` before submitting a patient intake. Model files being loadable
   at `/health/models` does not apply these database changes automatically.

## Export and install each notebook

Open the notebook in Jupyter and run every cell from top to bottom. The model
notebooks export these files under `MLModels/artifacts/`. When forecasting
selects Seasonal Naive, use the JSON export cell below because that baseline
has no fitted Statsmodels object.

| Bundle | Notebook output | Install command |
| --- | --- | --- |
| Severity | `severity_triage_model.joblib` | `python install_model_artifact.py severity Notebooks/artifacts/severity_triage_model.joblib --version severity-notebook-2026-08` |
| Stay duration | model + metadata Joblib files | `python install_model_artifact.py stay_duration Notebooks/artifacts/stay_duration_model.joblib --metadata Notebooks/artifacts/stay_duration_metadata.joblib --version stay-notebook-2026-08` |
| Transfer | model + metadata Joblib files | `python install_model_artifact.py transfer Notebooks/artifacts/transfer_model.joblib --metadata Notebooks/artifacts/transfer_metadata.joblib --version transfer-notebook-2026-08` |
| Discharge | model + metadata Joblib files | `python install_model_artifact.py discharge Notebooks/artifacts/discharge_outcome_model.joblib --metadata Notebooks/artifacts/discharge_outcome_metadata.joblib --version discharge-notebook-2026-08` |
| Forecast | `accident_forecast_seasonal_naive.json` | Keep the existing JSON until the forecasting notebook exports a replacement JSON artifact. |

The installed destinations are:

```text
app/ml_models/severity/severity_triage_model_notebook.joblib
app/ml_models/stay_duration/stay_duration_model_notebook.joblib
app/ml_models/stay_duration/stay_duration_metadata_notebook.joblib
app/ml_models/transfer/transfer_model_notebook.joblib
app/ml_models/transfer/transfer_metadata_notebook.joblib
app/ml_models/discharge/discharge_outcome_model_notebook.joblib
app/ml_models/discharge/discharge_outcome_metadata_notebook.joblib
app/ml_models/forecasting/accident_forecast_seasonal_naive.json
```

For the current Seasonal Naive winner, append this cell to the forecasting
notebook after `summary` has been created:

```python
from pathlib import Path
import json

artifacts = Path("artifacts")
artifacts.mkdir(exist_ok=True)

forecast_artifact = {
    "model_type": "seasonal_naive",
    "season_length": 12,
    "last_observed_period": monthly.index[-1].strftime("%Y-%m-%d"),
    "seasonal_values": [float(value) for value in monthly.iloc[-12:].values],
    "source_periods": [period.strftime("%Y-%m-%d") for period in monthly.index[-12:]],
    "validation": summary["held_out_test"],
}

output_path = artifacts / "accident_forecast_seasonal_naive.json"
with output_path.open("w", encoding="utf-8") as file:
    json.dump(forecast_artifact, file, indent=2)
print(f"Created: {output_path.resolve()}")
```

Do not copy the notebook itself into `app/ml_models`. Install only the exported
`.cbm`, `.joblib`, `.pkl`, or Seasonal Naive `.json` artifact.

## Verify the connection

Restart FastAPI after installing an artifact:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/health/models`. Each installed model should report
`"available": true`. The frontend will then replace `Unavailable` with its real
prediction automatically.

## If you retrain or change a notebook

The deployed artifact and `model_contract.json` must agree exactly on:

- final feature names and order;
- categorical versus numeric features;
- ordinal encodings;
- class labels;
- decision threshold;
- model format and artifact filename.

If the notebook selects a different model or feature set, update the matching
contract before restarting the API. Never change only the artifact while leaving
an old feature contract in place.

## Current contracts

- Stay duration predicts a bucket: `0-1 days`, `2-3 days`, or `4+ days`.
- Transfer predicts `Transfer`/`No transfer` and exposes the transfer probability.
- Discharge exposes all class probabilities for complete recovery, partial
  recovery, and further interventions.
- Forecasting returns monthly point forecasts. Confidence intervals are shown
  only for model types that produce them; Seasonal Naive does not.

Model errors, missing artifacts, and incomplete inputs are returned as explicit
`ERROR`, `UNAVAILABLE`, or `REFUSED` states. The frontend does not substitute
placeholder numbers.
