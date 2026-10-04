# Deployment configuration

How to configure, migrate, verify and test the Accident Care system before
deploying it by hand. Nothing in this document deploys anything, and every
value shown is a placeholder.

The system has two parts:

- **Backend:** this repository. FastAPI, served by uvicorn, using Supabase.
- **Frontend:** the sibling repository `Predictive-Dashboard---Accident-Care-Unit-Frontend`. A React/Vite single-page app.

---

## 1. Backend environment variables

Set these in the host's secret store, or in a local `.env` (git-ignored) for development. [.env.example](.env.example) is the template.

The API refuses to start when a required value is missing, is a placeholder or is unsafe. The startup error names the variable but never its value.

| Variable | Required | Secret | Format | If missing or invalid |
|---|---|---|---|---|
| `SECRET_KEY` | **yes** | **yes** | 32+ random characters, unique per environment | Startup fails (empty, placeholder, or shorter than 32 characters) |
| `SUPABASE_URL` | **yes** | no | `https://<project-ref>.supabase.co` | Startup fails (empty, placeholder or not a URL). Must be https in production |
| `SUPABASE_KEY` | **yes** | **yes** | Supabase **service-role** key | Startup fails (empty or placeholder) |
| `CORS_ORIGINS` | yes in production | no | Comma-separated exact origins, e.g. `https://<frontend-host>` | Development defaults to `http://localhost:5173,http://localhost:3000`. `*` is refused. Production refuses an empty list, `http://`, localhost and paths |
| `APP_ENV` | recommended | no | `development` or `production` | Defaults to `development`, and the startup log says so. Set `production` when deploying |
| `ALGORITHM` | no | no | `HS256` / `HS384` / `HS512` | Default `HS256`. Any other value fails startup |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | no | no | integer | Default 480 (an 8-hour shift) |
| `LOG_LEVEL` | no | no | `DEBUG` / `INFO` / `WARNING` / `ERROR` | Default `INFO` |
| `HF_MODEL_REPO` | no | no | `<owner>/<repo>` | Not set: the bundled `app/ml_models` is used |
| `HF_MODEL_REVISION` | no (recommended with `HF_MODEL_REPO`) | no | commit hash or tag | Not set: the latest revision at each start |
| `HF_TOKEN` | only for a private repository | **yes** | read-only Hugging Face token | The download fails and the bundled models are used (logged) |
| `HF_MODEL_CACHE_DIR` | no | no | writable directory | Default `/tmp/accident-care-models` |

Notes:

- **What `SECRET_KEY` signs.** It signs the API's own session tokens. It is not the Supabase JWT secret. Generate one with `python -c "import secrets; print(secrets.token_urlsafe(48))"`. Changing it signs every user out.
- **Why `SUPABASE_KEY` must be the service-role key:**
  - The admin "create user" flow calls the Supabase Auth admin API.
  - Every table has row level security with no policies, so only the service role can read or write them.
  - The key is used only by the backend. It is never returned by an endpoint and never sent to the frontend.
- **What `APP_ENV=production` changes:**
  - `/docs`, `/redoc` and `/openapi.json` are not served.
  - CORS must name exact https origins.
  - `SUPABASE_URL` must use https.
- **What the startup log contains:** the environment, the number of allowed CORS origins, the model source, and any model that is not ready. It never contains secret values. httpx request logging is kept at WARNING.

Test-only variables (`TEST_*`) are described in [section 8](#8-tests). They are never needed by the deployed API.

## 2. Frontend environment variables

Vite compiles **every `VITE_` variable into the JavaScript sent to browsers**, so none may hold a secret. The build refuses `VITE_` names that look like secrets (`SECRET`, `SERVICE_ROLE`, `PASSWORD`, `TOKEN`, `PRIVATE`, `SUPABASE_KEY`, `HF_`). See the frontend's `.env.example`.

| Variable | Where | Required | Format | If missing |
|---|---|---|---|---|
| `VITE_API_URL` | build time, shipped to browsers | yes for a separately hosted backend | `https://<backend-host>/api` | The app calls `/api` on its own origin. That is correct only behind a reverse proxy that serves the backend at `/api`. A production build warns, and **refuses a localhost value** unless `ALLOW_LOCAL_API_IN_BUILD=1` |
| `DEV_API_PROXY_TARGET` | dev server only | no | URL | `http://localhost:8000` |
| `DEV_ALLOWED_HOSTS` | dev server only | no | comma-separated hostnames, e.g. a tunnel | none |

The frontend talks only to this backend. It has no Supabase URL or key, and no other configuration.

## 3. Local development

```bash
# Backend
cp .env.example .env            # fill in SECRET_KEY and the development Supabase project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Frontend (separate terminal, in the frontend repository)
npm install
npm run dev                     # http://localhost:5173, proxies /api to port 8000
```

`--reload` is for development only. Run the deployed API without it, for example:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT --proxy-headers
```

## 4. Database migrations (manual, in this order)

Apply every file in `supabase/migrations/` in filename order, each in one transaction. For example, in the Supabase SQL editor, or with `psql -1 -v ON_ERROR_STOP=1 -f <file>`.

Every migration is written to be re-runnable. The test suite applies the whole chain twice to a fresh database on every run. Re-applying one that is already in production is therefore a no-op, apart from the notices described below.

| # | File | What it does | Notes |
|---|---|---|---|
| 1 | `202608010001_base_schema.sql` | The eleven original tables | `CREATE … IF NOT EXISTS`; no-op on the existing project |
| 2 | `202608190001_model_integration.sql` | Expanded intake and treatment columns | Additive |
| 3 | `202610030001_severity_input_columns.sql` | Severity input columns | Additive |
| 4 | `202610030002_integrity_constraints.sql` | Foreign keys, checks, unique staff email | **Preflight aborts with counts** if existing rows violate a new constraint; nothing is changed. Staff-without-hospital is `NOT VALID`, so the existing admin without a hospital is kept |
| 5 | `202610030003_injury_order.sql` | `injury_order` column | **Backfills** rows where it is NULL, inferring the order from the physical row order. A notice lists the episodes concerned |
| 6 | `202610030004_model_predictions.sql` | ML model, prediction and decision tables, with RLS | Additive |
| 7 | `202610030005_hospital_network.sql` | Seeds the eight network hospitals | Existing hospitals, including any outside the network, are unchanged |
| 8 | `202610040001_decision_hospital_scope.sql` | Decision trigger: hospital rules | Replaced again by #12 |
| 9 | `202610050001_audit_log_details.sql` | `AuditLog.details` (jsonb) | Additive |
| 10 | `202610060001_transfer_workflow.sql` | Transfer columns, state machine, one open transfer per episode, `transfer_transition()` | `EXECUTE` revoked from public, `anon` and `authenticated`; granted to `service_role` |
| 11 | `202610060002_original_tables_rls.sql` | RLS on the eleven original tables | No policies, so only the service role has access |
| 12 | `202610090001_decision_and_input_integrity.sql` | Decision finality, input locks, one decision per prediction | If existing data has duplicate decisions, the unique index is **not** created and a notice is raised; the trigger still blocks new duplicates |

**Dependencies:**
- #4 to #12 need the base tables (#1, #2 and #3).
- #8 replaces the decision trigger from #6, and #12 replaces #8.
- #10 and #12 rely on the network hospitals (#7) and the prediction tables (#6).
- The filename order satisfies all of these. The fresh-database build in the tests proves it.

**Destructive statements:**
- No migration drops or deletes data.
- #5 writes `injury_order` where it was NULL.
- #10's `UPDATE`s are inside the `transfer_transition()` function body; they run only when a transfer is processed.

The old root `supabase_model_integration.sql` was superseded by #2 and #3 and has been removed.

**After applying the migrations:**
1. Read the notices from #4, #5 and #12.
2. Check in the Supabase dashboard that RLS is on for all 16 tables.
3. Call `GET /health/database-schema`; it should return `"ready": true`.

## 5. Model configuration

**Where the models come from:**
- **Default:** the bundle in `app/ml_models/<model>/`. Each model has a `model_contract.json`, an artifact and its metadata.
- **Optional:** with `HF_MODEL_REPO`, the same folder layout is downloaded **once, at startup**, into `HF_MODEL_CACHE_DIR` and run locally.
  - If the download fails, or the repository lacks the expected folders, the bundled models stay active and the reason is logged.
  - Set `HF_MODEL_REVISION` to pin an exact commit.

**How models are loaded and identified:**
- **Loading:** artifacts are loaded on first use and cached per file. They are reloaded only if the file changes.
- **XAI:** uses the same model root and loader as inference.
- **Fingerprint:** every stored prediction records a SHA-256 fingerprint of its contract, artifact and metadata, so the exact model is always identifiable.

**Readiness:** `GET /health/models` shows `available`/`loadable` per model, with file names only. Models that aren't ready are also logged at startup.

**Unchanged by this configuration:** thresholds, labels and feature mappings. These come from the notebook bundles.

## 6. Authentication

- **Sign-in:** users sign in with Supabase Auth (email and password). The API then issues its own session token, signed with `SECRET_KEY` and valid for `ACCESS_TOKEN_EXPIRE_MINUTES`.
- **Role and hospital:** both are read from the database on every request, so a change takes effect on the next request.
- **Accounts an admin creates:** these carry a temporary password and must set their own at first sign-in (`/api/auth/change-password`). No email is sent.
- **Government officers:** live in `Government_Officer` and have no hospital.
- **Operator scripts:** `add_admin.py` and `add_gov.py` prompt for the password; they don't take it as an argument. Run them only with the intended project's service key.

## 7. CORS

- **Origins:** `CORS_ORIGINS` lists the exact browser origins allowed to call the API: scheme, host and optional port, with no path. For production, use the frontend's https origin. You can list more than one, comma-separated.
- **Credentials:** sessions travel in the `Authorization` header, so CORS credentials (cookies) are **disabled**.
- **Methods and headers:** only GET, POST and PATCH, with only the `Authorization` and `Content-Type` headers.
- **Exposed headers:** `X-Total-Count` is exposed for paged lists.

## 8. Tests

**Backend** (from this directory, with the virtual environment active):

```bash
pip install -r requirements-test.txt
python -m pytest tests            # unit + API + database tests
```

**What the test layers cover:**
- **Unit and API tests:** run anywhere.
- **Database tests:** need PostgreSQL server binaries (`initdb`, `pg_ctl`, `psql`) and are skipped without them.
  - They build a throwaway cluster from `supabase/migrations` only, applied twice.
  - The API-level tests also need Docker with `postgrest/postgrest:v12.2.3`.
  - `tests/db/test_local_stack_smoke.py` starts the **real** backend with `uvicorn` against that database. It covers sign-in, sessions, every role's workflow and the server log.
- **Never Supabase:** no test connects to Supabase.

**Live-backend browser tests.** Start the complete local stack:
- a throwaway database;
- PostgREST;
- an in-memory stand-in for Supabase Auth (`tests/local_stack`, test-only);
- the real backend on port 8000.

```bash
cp .env.test.example .env.test    # choose throwaway local credentials (git-ignored)
python -m tests.local_stack.run   # Ctrl-C stops it and deletes the database
```

Then, in the frontend repository, put the same `TEST_*` values in its `.env.test`, and run:

```bash
VITE_API_URL=/api npx playwright test
```

If your own backend or dev server is already running on port 8000 or 5173,
start the stack with `--port 8001` and run
`PLAYWRIGHT_PORT=5174 DEV_API_PROXY_TARGET=http://localhost:8001 VITE_API_URL=/api npx playwright test`.
The stack refuses to start on a busy port; without these overrides Playwright
would reuse your dev server, whose `/api` proxy points at your own backend.

Without the `TEST_*` variables, the live-backend specs fail at once with an explanation; the other specs run offline.

**Frontend checks:** `npx tsc -b`, `npm run lint`, `npm run build`.

**Load testing (optional, development only):**
1. `python fetch_tokens.py` writes the git-ignored `tokens.json` from the `TEST_*` accounts.
2. Run `locust`.
3. Delete `tokens.json` afterwards.

**Development test accounts:** `create_test_users.py` provisions the `TEST_*` accounts in a **development** Supabase project.
- It needs `--confirm-host <project host>` and `TEST_HOSPITAL_NAME`.
- It never creates hospitals.

## 9. Manual deployment checklist

**Secrets**
- [ ] Generate a new `SECRET_KEY` for production (32+ random characters) and store it only in the host's secret store.
- [ ] Set `SUPABASE_URL` (https) and `SUPABASE_KEY` (service role), plus `HF_TOKEN` if you use a private model repository.
- [ ] Set `APP_ENV=production` and `CORS_ORIGINS=https://<frontend-host>`.
- [ ] Build the frontend with `VITE_API_URL=https://<backend-host>/api`, or serve the backend at `/api` on the frontend host.

**Credentials to revoke or rotate.** These have been committed to git history:
- [ ] **The backend `.env`.** It was committed on 2026-08-02 and removed in the next commit. It is still in the history of the GitHub repository. Rotate everything it held: the Supabase service-role key (and the JWT secret, if it was there), `SECRET_KEY`, and any Hugging Face token.
- [ ] **`tokens.json`.** It was committed on 2026-09-26. The session tokens in it have expired (8-hour lifetime), but they were signed with the `SECRET_KEY` of that time, so rotate that key as well.
- [ ] **The demo password.** It appears in the history of both repositories. Delete the demo accounts (`nurse@`, `doctor@`, `admin@` and `gov@accidentcare.org`) or change their passwords in every Supabase project where they exist.
- [ ] **History rewrite (optional).** Removing these from history would need a separate, manual history rewrite and force-push, for example with `git filter-repo`. Rotating the secrets is what makes them harmless.

**Database**
- [ ] Back up the production database.
- [ ] Apply the migrations in [section 4](#4-database-migrations-manual-in-this-order) in order, and keep the notices.
- [ ] Check in the Supabase dashboard that RLS is on for all 16 tables. Optionally, with the project's anon key, confirm `GET /rest/v1/Patient` returns no rows.
- [ ] Decide what to do with the production admin who has no hospital. Until they are assigned one, the API refuses them every staff endpoint.
- [ ] Decide what to do with the staff accounts at General Hospital Colombo, which is outside the eight-hospital network: patients cannot be transferred to Colombo, though its admins can still request transfers out to network hospitals. Move them deliberately, or leave them.
- [ ] If #12 reports duplicate decisions, review those episodes.

**Supabase**
- [ ] Auth: email/password sign-in on, and email confirmation as you require. Accounts created by admins are pre-confirmed.
- [ ] Keep public sign-ups disabled if accounts should be created only by admins.
- [ ] Keep the service-role key out of every client application.

**After deploying**
- [ ] `GET /health`, then `/health/models` (all `loadable`), then `/health/database-schema` (`ready: true`).
- [ ] `/docs` returns 404 in production.
- [ ] Sign in with each role using real, non-demo accounts. Government officers see only Forecast and Association Rules.

## 10. Known limitations, unchanged by design

- **Severity LOW cannot occur.** No input reaches the LOW zone with the deployed Severity model; the threshold is unchanged.
- **Decisions are final.** There is no correction workflow.
- **No live notifications and no email delivery.** Temporary passwords are shown to the creating admin once.
- **Sign-in matches by email.** The profile is linked by email, not by the Supabase Auth user id.
- **Age is shared.** Age belongs to the patient, so all their episodes share it.
- **Treatment days are summed.** Multiple treatment rows add up their days (per-ward stays).
- **Unreferenced model artifacts are kept.** Some older model artifacts remain in the repository on purpose.
