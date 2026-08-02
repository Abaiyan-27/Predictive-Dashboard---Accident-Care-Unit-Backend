# Accident Care Unit — Predictive Dashboard Backend

FastAPI REST backend for the Accident Care hospital dashboard with JWT auth, RBAC, SQLite persistence, and ML-based patient triage.

---

## Quick Start (Manual Setup)

### 1. Prerequisites
- Python 3.10 or higher
- pip

### 2. Create & Activate a Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux / macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now live at **http://localhost:8000**

- **Interactive API Docs (Swagger UI):** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health check:** http://localhost:8000/health

On first startup, the database (`accident_care.db`) is created automatically and seeded with demo accounts and sample patients.

---

## Demo Accounts

| Role | Email | Password |
|------|-------|----------|
| Nurse | `nurse@accidentcare.org` | `demo` |
| Doctor | `doctor@accidentcare.org` | `demo` |
| Admin | `admin@accidentcare.org` | `demo` |
| Government | `gov@accidentcare.org` | `demo` |

---

## Project Structure

```
backend/
├── app/
│   ├── main.py              ← FastAPI app, CORS, router registration
│   ├── core/
│   │   ├── config.py        ← Pydantic-settings (reads .env)
│   │   └── security.py      ← bcrypt hashing + JWT creation/decode
│   ├── db/
│   │   ├── base.py          ← SQLAlchemy DeclarativeBase
│   │   ├── session.py       ← Engine + get_db() dependency
│   │   └── init_db.py       ← Table creation + seed data
│   ├── models/              ← SQLAlchemy ORM models
│   │   ├── user.py
│   │   ├── patient.py
│   │   ├── hospital.py
│   │   ├── transfer.py
│   │   └── audit_log.py
│   ├── schemas/             ← Pydantic v2 request/response schemas
│   │   ├── auth.py
│   │   ├── user.py
│   │   ├── patient.py
│   │   ├── hospital.py
│   │   ├── transfer.py
│   │   ├── audit_log.py
│   │   └── analytics.py
│   ├── api/                 ← Route handlers
│   │   ├── deps.py          ← get_current_user, RoleChecker
│   │   ├── auth.py          ← POST /api/auth/login
│   │   ├── nurse.py         ← POST /api/patients, GET /api/patients/my
│   │   ├── doctor.py        ← GET /api/patients/{id}, /prognosis, /hospital/{id}
│   │   ├── admin.py         ← /api/admin/* endpoints
│   │   └── government.py    ← /api/government/* endpoints
│   └── services/
│       ├── ml_service.py    ← Heuristic ML engine (swap-in ready for real models)
│       └── audit_service.py ← Audit log helper
├── requirements.txt
└── .env
```

---

## API Endpoints

### Auth
| Method | Path | Access |
|--------|------|--------|
| POST | `/api/auth/login` | Public |

### Nurse / Patients
| Method | Path | Roles |
|--------|------|-------|
| POST | `/api/patients` | nurse, admin |
| GET | `/api/patients/my` | any authenticated |

### Doctor / Patients
| Method | Path | Roles |
|--------|------|-------|
| GET | `/api/patients/hospital/{hospital_id}` | doctor, admin |
| GET | `/api/patients/{patient_id}` | nurse, doctor, admin |
| GET | `/api/patients/{patient_id}/prognosis` | doctor |

### Admin
| Method | Path | Roles |
|--------|------|-------|
| GET | `/api/admin/hospital-status` | admin |
| POST | `/api/admin/users` | admin |
| GET | `/api/admin/transfers` | admin, doctor |
| PATCH | `/api/admin/transfers/{id}/status` | admin |
| GET | `/api/admin/audit-logs` | admin |

### Government
| Method | Path | Roles |
|--------|------|-------|
| GET | `/api/government/accident-trends` | government |
| GET | `/api/government/risk-predictions` | government |

---

## Switching to PostgreSQL

Change `DATABASE_URL` in `.env`:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/accident_care
```

Then install the driver:
```bash
pip install psycopg2-binary
```

No other code changes are needed — SQLAlchemy handles the dialect switch automatically.

---

## Connecting the Frontend

The frontend (`http://localhost:5173`) is already allowlisted in CORS. To switch from mock data to this real backend, update the API layer in the frontend — all function signatures and response shapes match the TypeScript interfaces exactly.