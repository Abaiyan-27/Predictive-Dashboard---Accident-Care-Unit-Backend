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
│  6. Database Layer (app/db/ & app/models/)                                  │
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
* **[.env](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/.env)**: Environment configuration for JWT secret, token expiration time, database URL, and CORS origins.
* **[requirements.txt](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/requirements.txt)**: Python package manifest.
* **[README.md](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/README.md)**: Quick start guide and commands.

---

### 📁 `app/` Directory

#### Application Entry Point
* **[app/main.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/main.py)**
  * Initializes the `FastAPI` instance.
  * Configures `CORSMiddleware` to allow requests from the React frontend.
  * Registers all routers with the `/api` prefix.
  * Manages the application lifecycle (`lifespan`) to auto-create and seed the database on initial launch.
  * Provides a `/health` endpoint for monitoring.

---

#### Core Utilities (`app/core/`)
* **[app/core/config.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/core/config.py)**
  * Uses `pydantic-settings` to parse configuration variables from `.env` or system environment.
  * Implements `get_settings()` with `@lru_cache` to avoid reloading configuration on every request.
* **[app/core/security.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/core/security.py)**
  * `get_password_hash(plain_pwd)`: Hashes passwords with bcrypt.
  * `verify_password(plain_pwd, hashed_pwd)`: Timing-safe password verification.
  * `create_access_token(data, expires_delta)`: Creates signed JWT tokens with UTC expiration.
  * `decode_token(token)`: Validates JWT signature and extracts the payload.

---

#### Database Layer (`app/db/`)
* **[app/db/base.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/db/base.py)**
  * Declares the SQLAlchemy `Base(DeclarativeBase)` class inherited by all ORM entities.
* **[app/db/session.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/db/session.py)**
  * Initializes the database `engine` (with SQLite thread-safety settings if applicable).
  * Provides the `get_db()` dependency generator that supplies a scoped session to endpoints and guarantees cleanup (`db.close()`).
* **[app/db/init_db.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/db/init_db.py)**
  * Runs on startup: builds tables and seeds demo accounts (`nurse`, `doctor`, `admin`, `gov`), hospitals, demo patients, transfers, and initial audit logs.
  * Uses `db.query(User).count() == 0` check to ensure idempotent startup without creating duplicates.

---

#### Database Models (`app/models/`)
* **[app/models/user.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/models/user.py)**: `User` entity (`id`, `name`, `email`, `hashed_password`, `role`, `hospital_id`, `hospital_name`).
* **[app/models/patient.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/models/patient.py)**: `Patient` entity storing intake information (`age`, `gender`, `collision_type`, `injury_description`) along with ML-predicted results (`severity`, `priority`, `predicted_los_days`).
* **[app/models/hospital.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/models/hospital.py)**: `Hospital` entity tracking bed capacities (`total_beds`, `occupied_beds`, `available_beds`).
* **[app/models/transfer.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/models/transfer.py)**: `Transfer` entity for hospital-to-hospital transfer requests (`source_hospital`, `destination_hospital`, `status`).
* **[app/models/audit_log.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/models/audit_log.py)**: `AuditLog` entity recording user actions with timestamp for compliance.

---

#### Pydantic Schemas (`app/schemas/`)
* **[app/schemas/auth.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/schemas/auth.py)**: `LoginCredentials`, `AuthResponse`.
* **[app/schemas/user.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/schemas/user.py)**: `UserOut`, `CreateUserPayload`.
* **[app/schemas/patient.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/schemas/patient.py)**: `PatientIntakeForm`, `PatientOut`, `PrognosisOut`.
* **[app/schemas/hospital.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/schemas/hospital.py)**: `HospitalStatus`.
* **[app/schemas/transfer.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/schemas/transfer.py)**: `TransferOut`, `UpdateTransferStatus`.
* **[app/schemas/audit_log.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/schemas/audit_log.py)**: `AuditLogOut`.
* **[app/schemas/analytics.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/schemas/analytics.py)**: `AccidentTrend`, `RiskPrediction`.

---

#### Business & ML Services (`app/services/`)
* **[app/services/ml_service.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/services/ml_service.py)**
  * Encapsulates all prediction logic.
  * Contains a realistic heuristic fallback engine with stable signatures.
  * Ready to load real ML models (`.joblib` / `.pkl`) for:
    1. Severity prediction (`Critical`, `Moderate`, `Minor`)
    2. Triage priority (`High`, `Medium`, `Low`)
    3. Length of Stay (`predicted_los_days`)
    4. Recovery level (`Full Recovery`, `Partial Recovery`, `Long-term Care`)
    5. Partial recovery probability (`0-100%`)
    6. Regional accident risk forecasting
* **[app/services/audit_service.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/services/audit_service.py)**
  * Helper function `log_action()` that writes an audit entry and calls `db.flush()` so audit records remain atomic with the business transaction.

---

#### API Routers & Dependencies (`app/api/`)
* **[app/api/deps.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/api/deps.py)**:
  * `get_current_user`: Decodes JWT from Authorization Bearer header, validates user existence in DB.
  * `RoleChecker`: Callable class for role validation. Provides pre-built dependencies: `require_nurse`, `require_doctor`, `require_admin`, `require_government`.
* **[app/api/auth.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/api/auth.py)**: Login route (`POST /api/auth/login`).
* **[app/api/nurse.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/api/nurse.py)**: Patient intake (`POST /api/patients`), viewing nurse-admitted patients (`GET /api/patients/my`).
* **[app/api/doctor.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/api/doctor.py)**: Hospital patient roster, patient details, and on-demand prognosis prediction (`GET /api/patients/{id}/prognosis`).
* **[app/api/admin.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/api/admin.py)**: Real-time bed status, user creation, transfer management, and audit log inspection.
* **[app/api/government.py](file:///home/yusri-ahamed/Sem_05/DSEProject/Ours/Predictive-Dashboard---Accident-Care-Unit-Backend/app/api/government.py)**: 12-month aggregated accident trends and regional risk analytics (no PII).

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
       │ severity (ML)          │        └────────────────────────┘
       │ priority (ML)          │
       │ predicted_los_days(ML) │        ┌────────────────────────┐
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

### The 6 Prediction Capabilities:
1. **Severity Prediction**: Assigns `Critical`, `Moderate`, or `Minor` based on injury description, collision dynamics, and age vulnerability.
2. **Triage Priority**: Assigns `High`, `Medium`, or `Low` priority for emergency room queuing.
3. **Length of Stay (LOS)**: Estimates number of days the patient will occupy an inpatient bed.
4. **Expected Recovery Level**: Forecasts `Full Recovery`, `Partial Recovery`, or `Long-term Care`.
5. **Partial Recovery Probability**: Calculates a numerical likelihood percentage (`0%–100%`).
6. **Regional Accident Forecasting**: Projects risk levels and incident frequencies across urban sectors.

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
1. **Nurse submits form** via frontend (`name`, `age`, `gender`, `collision_type`, `injury_description`).
2. Frontend calls `POST /api/patients` with `Authorization: Bearer <token>`.
3. `require_nurse` validates that the token belongs to a user with the `nurse` role.
4. `ml_service.predict_severity()` analyzes the clinical text and calculates:
   * `severity`: `Critical`
   * `priority`: `High`
   * `predicted_los_days`: `15`
5. Patient record is persisted to the database.
6. `audit_service.log_action()` records: `"Patient 'Marcus Chen' admitted. Severity: Critical, Priority: High, LOS: 15 days."`
7. Response is returned to the frontend and rendered in the dashboard list.

### Scenario B: Doctor Prognosis Review
1. **Doctor selects an admitted patient** on the dashboard.
2. Frontend calls `GET /api/patients/{patient_id}/prognosis`.
3. `require_doctor` verifies the user's role.
4. Backend retrieves stored patient attributes and passes them to `ml_service.predict_prognosis()`.
5. Returns `expected_recovery_level`, `partial_recovery_probability`, and transfer suggestions.

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
