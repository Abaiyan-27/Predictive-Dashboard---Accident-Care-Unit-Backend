# Accident Care Dashboard — Backend

FastAPI REST backend using Supabase for authentication and persistence, custom
JWTs for API sessions, role-based access control, and CatBoost severity
prediction.

## Configure

Create `.env` with:

```env
SECRET_KEY=replace-with-a-long-random-secret
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=480
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
CORS_ORIGINS=http://localhost:5173
```

`SUPABASE_KEY` must be a service-role key because the administrator user flow
creates Supabase Auth accounts. Keep `.env` private.

### Optional: models stored on Hugging Face

The backend always runs predictions locally. To store the model bundle in a
private Hugging Face repository and download it when FastAPI starts, add these
values to `.env`:

```env
HF_MODEL_REPO=yusri009/accident-care-models
HF_TOKEN=your-read-token
HF_MODEL_CACHE_DIR=/tmp/accident-care-models
```

Do not commit the token. The Hugging Face repository must have the same root
folders as `app/ml_models` (`severity`, `stay_duration`, `transfer`,
`discharge`, and `forecasting`). If Hugging Face cannot be reached, the bundled
local models remain the fallback.

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: <http://localhost:8000/health>
- Swagger: <http://localhost:8000/docs>

## Optional demo accounts

To provision all four roles in the configured Supabase project, run:

```bash
python create_test_users.py
```

The script is idempotent and prints the demo credentials when complete. It
creates or updates external Supabase data, so run it only against the intended
development project.

## Main API groups

- `/api/auth` — login
- `/api/patients` and `/api/accidents` — intake and clinical records
- `/api/treatments` — doctor treatment logging
- `/api/admin` — hospitals, staff, transfers, and audit logs
- `/api/government` — real accident trends, model forecasts, and statistics

## Model artifacts

See [MODEL_INTEGRATION.md](MODEL_INTEGRATION.md) for the notebook export,
installation, Supabase migration, and verification workflow.
