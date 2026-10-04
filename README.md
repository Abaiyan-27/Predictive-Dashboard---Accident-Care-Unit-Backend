# Accident Care Dashboard — Backend

FastAPI REST backend using Supabase for authentication and persistence, custom
JWTs for API sessions, role-based access control, and CatBoost severity
prediction.

## Configure

Copy [.env.example](.env.example) to `.env` (git-ignored) and fill it in.
`SUPABASE_KEY` must be the service-role key, because the administrator user
flow creates Supabase Auth accounts; keep it server-side only. The API refuses
to start with a missing, placeholder or short `SECRET_KEY`.

Every variable, the migration order, CORS, model configuration, the test setup
and the manual deployment checklist are in [DEPLOYMENT.md](DEPLOYMENT.md).

### Optional: models stored on Hugging Face

The backend always runs predictions locally. To download the model bundle from
a Hugging Face repository once at startup, set `HF_MODEL_REPO`,
`HF_MODEL_REVISION` (a commit hash, so every start runs the same artifacts)
and, for a private repository, `HF_TOKEN`. The repository must have the same
root folders as `app/ml_models` (`severity`, `stay_duration`, `transfer`,
`discharge` and `forecasting`). If Hugging Face cannot be reached, the bundled
local models remain the fallback.

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Health: <http://localhost:8000/health>
- Swagger: <http://localhost:8000/docs>

## Optional test accounts (development projects only)

`create_test_users.py` provisions one account per role from the `TEST_*`
variables (see [.env.test.example](.env.test.example)); no password is stored
in the repository. It resets the password of existing accounts with those
emails, so it requires the target host to be typed out:

```bash
python create_test_users.py --confirm-host <project-ref>.supabase.co
```

Never run it against production.

## Main API groups

- `/api/auth` — login
- `/api/patients` and `/api/accidents` — intake and clinical records
- `/api/treatments` — doctor treatment logging
- `/api/admin` — hospitals, staff, transfers, and audit logs
- `/api/government` — real accident trends, model forecasts, and statistics

## Model artifacts

See [MODEL_INTEGRATION.md](MODEL_INTEGRATION.md) for the notebook export,
installation, Supabase migration, and verification workflow.
