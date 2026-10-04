# 🚑 Accident Care Unit — Backend Architecture & Complete Guide

> **Historical document:** this guide describes the original local-database and
> heuristic prototype. The current application uses Supabase and contract-driven
> model artifacts. Use `README.md` and `MODEL_INTEGRATION.md` for current setup.

A comprehensive architectural and technical explanation of the **FastAPI** backend for the **Accident Care** dashboard.

---

## 📑 Table of Contents
1. [System Architecture & Request Lifecycle](#1-system-architecture--request-lifecycle)
2. [Tech Stack & Design Decisions](#2-tech-stack--design-decisions)
3. [Directory & File Walkthrough](#3-directory--file-walkthrough)
4. [Authentication & Role-Based Access Control (RBAC)](#4-authentication--role-based-access-control-rbac)
5. [Database Layer & Data Models](#5-database-layer--data-models)
6. [Machine Learning (ML) & Predictive Engine](#6-machine-learning-ml--predictive-engine)
7. [API Endpoints & Access Matrix](#7-api-endpoints--access-matrix)
8. [End-to-End Data Flow Examples](#8-end-to-end-data-flow-examples)
9. [How to Run & Verify](#9-how-to-run--verify)

---

## 1. System Architecture & Request Lifecycle

The backend adopts a clean, layered architecture separating routing, business logic, validation, and data persistence:

```
[ React + TypeScript Frontend ] (http://localhost:5173)
              │
              ▼  (HTTP / JSON Requests with Authorization: Bearer <JWT>)
┌─────────────────────────────────────────────────────────────────────────────┐
│                             FastAPI Application                             │
│                                                                             │
│  1. CORS Middleware (Validates Origin & Headers)                            │
│  2. Routers (/api/auth, /api/patients, /api/admin, /api/government)         │
│  3. Dependencies Layer (app/api/deps.py)                                    │
│     ├── Extracts & verifies JWT token                                       │
│     ├── Fetches authenticated User from DB                                  │
│     └── RBAC RoleChecker (Nurse, Doctor, Admin, Government)                 │
│                                                                             │
│  4. Schemas Layer (app/schemas/)                                            │
│     └── Pydantic v2 validates request payload & formats response JSON       │
│                                                                             │
│  5. Services Layer (app/services/)                                          │
│     ├── ml_service.py    ──▶ Runs Severity, LOS, Prognosis & Risk Inference │
│     └── audit_service.py ──▶ Logs user actions for compliance               │
│                                                                             │
│  6. Database Layer (app/db/ & supabase/migrations/)                        │
│     └── SQLAlchemy 2.0 ORM translates python objects into SQL queries       │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
   [ SQLite / PostgreSQL DB ]
```

---

## 2. Tech Stack & Design Decisions

| Technology | Purpose | Why It Was Chosen |
|---|---|---|
| **FastAPI** | REST API Framework | High performance (async ready), automatic OpenAPI/Swagger documentation, native type hinting. |
| **Pydantic v2** | Data Validation | Fast schema validation, strict typing, seamless serialization matching frontend TypeScript types. |
| **SQLAlchemy 2.0** | ORM (Object Relational Mapping) | Clean declarative syntax (`Mapped`, `mapped_column`), cross-database compatibility (SQLite for dev, PostgreSQL for prod). |
| **python-jose** | JWT Authentication | Encodes user ID and Role directly in the token payload to minimize database lookups. |
| **passlib + bcrypt** | Password Hashing | Cryptographically secure one-way hashing with salt protection. |
| **Uvicorn** | ASGI Web Server | Lightweight, lightning-fast ASGI server for production and local development with hot reloading. |

---

## 3. Directory & File Walkthrough

### 📁 Root Directory
* **.env** (git-ignored; template [.env.example](.env.example)): environment configuration for the session signing key, token expiration time, Supabase URL and service-role key, and CORS origins. See [DEPLOYMENT.md](DEPLOYMENT.md).
* **[requirements.txt](requirements.txt)**: Python package manifest.
* **[README.md](README.md)**: Quick start guide and commands.

---

### 📁 `app/` Directory

#### Application Entry Point
* **[app/main.py](app/main.py)**
  * Initializes the `FastAPI` instance.
  * Configures `CORSMiddleware` to allow requests from the React frontend.
  * Registers all routers with the `/api` prefix.
  * Manages the application lifecycle (`lifespan`) to auto-create and seed the database on initial launch.
  * Provides a `/health` endpoint for monitoring.

---

#### Core Utilities (`app/core/`)
* **[app/core/config.py](app/core/config.py)**
  * Uses `pydantic-settings` to parse configuration variables from `.env` or system environment.
  * Implements `get_settings()` with `@lru_cache` to avoid reloading configuration on every request.
* **[app/core/security.py](app/core/security.py)**
  * `get_password_hash(plain_pwd)`: Hashes passwords with bcrypt.
  * `verify_password(plain_pwd, hashed_pwd)`: Timing-safe password verification.
  * `create_access_token(data, expires_delta)`: Creates signed JWT tokens with UTC expiration.
  * `decode_token(token)`: Validates JWT signature and extracts the payload.

---

#### Database Layer (`app/db/`)
* **[app/db/base.py](app/db/base.py)**
  * Declares the SQLAlchemy `Base(DeclarativeBase)` class inherited by all ORM entities.
* **[app/db/session.py](app/db/session.py)**
  * Initializes the database `engine` (with SQLite thread-safety settings if applicable).
  * Provides the `get_db()` dependency generator that supplies a scoped session to endpoints and guarantees cleanup (`db.close()`).
* **[app/db/init_db.py](app/db/init_db.py)**
  * Runs on startup: builds tables and seeds demo accounts (`nurse`, `doctor`, `admin`, `gov`), hospitals, demo patients, transfers, and initial audit logs.
  * Uses `db.query(User).count() == 0` check to ensure idempotent startup without creating duplicates.

---

#### Database schema (`supabase/migrations/`)
The schema is defined by the SQL migrations, applied in filename order. The old `app/models/` classes were removed in Phase 9: nothing imported them, and the API reads and writes rows through Supabase directly. Model outputs are stored in `Model_Prediction` and human decisions in `Prediction_Decision`.

#### Pydantic Schemas (`app/schemas/`)
* **[app/schemas/auth.py](app/schemas/auth.py)**: `LoginCredentials`, `AuthResponse`.
* **[app/schemas/user.py](app/schemas/user.py)**: `UserOut`, `CreateUserPayload`.
* **[app/schemas/patient.py](app/schemas/patient.py)**: `PatientIntakeForm`, `PatientOut`, `PrognosisOut`.
* **[app/schemas/hospital.py](app/schemas/hospital.py)**: `HospitalStatus`.
* **[app/schemas/transfer.py](app/schemas/transfer.py)**: `TransferOut`, `UpdateTransferStatus`.
* **[app/schemas/audit_log.py](app/schemas/audit_log.py)**: `AuditLogOut`.
* **[app/schemas/analytics.py](app/schemas/analytics.py)**: `AccidentTrend`, `RiskPrediction`.

---

#### Business & ML Services (`app/services/`)
* **[app/services/ml_service.py](app/services/ml_service.py)**
  * Encapsulates all prediction logic.
  * Runs the four deployed patient models and returns only their contract labels:
    1. Severity (nurse decision): `HIGH`, `BORDERLINE`, `LOW`, or `DEFER TO CLINICAL JUDGEMENT` when it cannot score
    2. Stay Duration (doctor decision): `0-1 days`, `2-3 days`, `4+ days`
    3. Transfer (doctor decision): `Transfer`, `No transfer`
    4. Discharge Outcome (doctor decision): `Complete recovery`, `Partial recovery`, `Further interventions`
  * Also serves the seasonal-naive monthly accident forecast for Government (its Low / Medium / High bands describe monthly accident volume, not patient Severity).
* **[app/services/audit_service.py](app/services/audit_service.py)**
  * Helper function `log_action()` that writes an audit entry and calls `db.flush()` so audit records remain atomic with the business transaction.

---

#### API Routers & Dependencies (`app/api/`)
* **[app/api/deps.py](app/api/deps.py)**:
  * `get_current_user`: Decodes JWT from Authorization Bearer header, validates user existence in DB.
  * `RoleChecker`: Callable class for role validation. Provides pre-built dependencies: `require_nurse`, `require_doctor`, `require_admin`, `require_government`.
* **[app/api/auth.py](app/api/auth.py)**: Login route (`POST /api/auth/login`).
* **[app/api/nurse.py](app/api/nurse.py)**: Patient intake (`POST /api/patients`), viewing nurse-admitted patients (`GET /api/patients/my`).
* **[app/api/doctor.py](app/api/doctor.py)**: Hospital patient roster, patient details, and on-demand prognosis prediction (`GET /api/patients/{id}/prognosis`).
* **[app/api/admin.py](app/api/admin.py)**: Real-time bed status, user creation, transfer management, and audit log inspection.
* **[app/api/government.py](app/api/government.py)**: 12-month aggregated accident trends and regional risk analytics (no PII).

---

## 4. Authentication & Role-Based Access Control (RBAC)

### 1. The 4 Roles
1. **`nurse`**: Submits patient intake, views patients they admitted.
2. **`doctor`**: Views admitted patients, inspects full records, requests ML prognosis.
3. **`admin`**: Monitors hospital bed capacity, creates new staff users, approves/rejects transfers, inspects audit logs.
4. **`government`**: Accesses macro analytics (monthly trends, regional predictions) without PII.

### 2. How the RBAC Guard Works
When a request hits a protected endpoint:
1. `get_current_user` extracts the JWT from the `Authorization: Bearer <token>` header.
2. It verifies the signature and retrieves the user object.
3. `RoleChecker(["allowed_roles"])` checks if `current_user.role` is in the permitted list.
4. If unauthorized $\rightarrow$ Returns `401 Unauthorized` (invalid token) or `403 Forbidden` (insufficient role permission).

---

## 5. Database Layer & Data Models

### Entity Relationship Diagram

```
       ┌────────────────────────┐
       │         User           │
       ├────────────────────────┤
       │ id (PK)                │
       │ name                   │
       │ email (Unique)         │
       │ hashed_password        │
       │ role                   │
       │ hospital_id            │
       │ hospital_name          │
       └───────────┬────────────┘
                   │ 1
                   │ creates
                   │ *
       ┌───────────▼────────────┐        ┌────────────────────────┐
       │        Patient         │        │        Hospital        │
       ├────────────────────────┤        ├────────────────────────┤
       │ id (PK)                │        │ id (PK)                │
       │ name, age, gender      │        │ name                   │
       │ admission_date         │        │ total_beds             │
       │ collision_type         │        │ occupied_beds          │
       │ injury_description     │        │ available_beds         │
       │ (historical diagram)   │        └────────────────────────┘
       │ see supabase/migrations│
       │ for the current schema │        ┌────────────────────────┐
       │ created_by (FK: users) │        │        Transfer        │
       │ hospital_id            │        ├────────────────────────┤
       └────────────────────────┘        │ id (PK)                │
                                         │ patient_name           │
       ┌────────────────────────┐        │ source_hospital        │
       │       AuditLog         │        │ destination_hospital   │
       ├────────────────────────┤        │ status                 │
       │ id (PK)                │        │ requested_date         │
       │ user, role, action     │        └────────────────────────┘
       │ timestamp (ISO 8601)   │
       └────────────────────────┘
```

---

## 6. Machine Learning (ML) & Predictive Engine

All predictive features are centralized in `app/services/ml_service.py`.

### The prediction capabilities
1. **Severity** (run by the nurse at intake): `HIGH`, `BORDERLINE` or `LOW`; `DEFER TO CLINICAL JUDGEMENT` when the model cannot score the patient, in which case the nurse makes the clinical decision. The label list lives in `app/contracts/severity_labels.json`.
2. **Stay Duration**: `0-1 days`, `2-3 days`, `4+ days`.
3. **Transfer**: `Transfer`, `No transfer`. A prediction is not a transfer request; requests follow the admin → nurse workflow.
4. **Discharge Outcome**: `Complete recovery`, `Partial recovery`, `Further interventions`.
5. **Accident forecast** (Government): a seasonal-naive baseline of monthly accident counts, not a patient model.

Every prediction is stored append-only in `Model_Prediction`; the nurse's or doctor's final decision is stored separately in `Prediction_Decision`. See `MODEL_INTEGRATION.md` for the artifacts.

### How to plug in trained models:
Place `.joblib` or `.pkl` files in `app/ml_models/` and load them inside `ml_service.py` using `joblib.load()`. The API routers and frontend do not need any code changes because the return signatures remain consistent.

---

## 7. API Endpoints & Access Matrix

| Endpoint | Method | Allowed Roles | Description |
|---|---|---|---|
| `/api/auth/login` | `POST` | Public | Authenticates credentials and returns JWT token + user details |
| `/api/patients` | `POST` | `nurse`, `admin` | Submits patient intake, triggers ML severity/LOS, logs audit event |
| `/api/patients/my` | `GET` | Authenticated | Fetches patients admitted by the current user |
| `/api/patients/hospital/{hospital_id}` | `GET` | `doctor`, `admin` | Lists all patients in a specific hospital |
| `/api/patients/{patient_id}` | `GET` | `nurse`, `doctor`, `admin` | Fetches a single patient profile |
| `/api/patients/{patient_id}/prognosis` | `GET` | `doctor` | Runs ML prognosis inference for a patient |
| `/api/admin/hospital-status` | `GET` | `admin` | Real-time bed occupancy stats synced with patient count |
| `/api/admin/users` | `POST` | `admin` | Creates new staff accounts (default password: `demo`) |
| `/api/admin/transfers` | `GET` | `admin`, `doctor` | Lists all inter-hospital transfer requests |
| `/api/admin/transfers/{id}/status` | `PATCH` | `admin` | Approves or rejects a patient transfer |
| `/api/admin/audit-logs` | `GET` | `admin` | Inspects system activity history |
| `/api/government/accident-trends` | `GET` | `government` | Returns 12-month aggregated accident trends |
| `/api/government/risk-predictions` | `GET` | `government` | Returns regional risk and incident predictions |
| `/health` | `GET` | Public | Service health probe |

---

## 8. End-to-End Data Flow Examples

### Scenario A: Patient Intake Flow (Nurse)
1. **Nurse saves the basic details** (`POST /api/patients/intake`), then the Severity inputs (`PATCH /api/patients/intake/{accident_id}`). Each save is confirmed in the UI first.
2. The nurse runs Severity (`POST /api/severity/accidents/{accident_id}/run`); the prediction is stored in `Model_Prediction`.
3. The nurse accepts the prediction or records their own decision (`POST /api/severity/predictions/{prediction_id}/decision`). A `DEFER TO CLINICAL JUDGEMENT` result always needs the nurse's clinical decision.
4. After the decision the Severity inputs are locked, and the event is written to the audit log.

### Scenario B: Doctor Review
1. **Doctor opens a patient** from the hospital list (`GET /api/doctor/accidents/{accident_id}`): Severity is shown read-only with the nurse's decision.
2. The doctor records treatment (`POST /api/treatments`); the Stay, Transfer and Discharge models run on it.
3. For each model the doctor accepts the prediction or records their own decision (`POST /api/doctor/accidents/{accident_id}/predictions/{prediction_id}/decision`).
4. Out-of-date predictions (inputs changed, or the patient moved hospital) must be run again before a decision.

---

## 9. How to Run & Verify

```bash
# 1. Activate your virtual environment
source venv/bin/activate

# 2. Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

* **Interactive Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

### Seed Credentials for Testing:
* Nurse: `nurse@accidentcare.org` / `demo`
* Doctor: `doctor@accidentcare.org` / `demo`
* Admin: `admin@accidentcare.org` / `demo`
* Government: `gov@accidentcare.org` / `demo`
