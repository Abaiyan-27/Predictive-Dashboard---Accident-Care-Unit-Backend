"""
Database initialiser — creates tables and seeds demo data.
Called once on application startup.
"""
from __future__ import annotations

import uuid
from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.core.security import get_password_hash
from app.db.base import Base
from app.db.session import engine, SessionLocal
from app.models.audit_log import AuditLog
from app.models.hospital import Hospital
from app.models.patient import Patient
from app.models.transfer import Transfer
from app.models.user import User

# ── Seed data ────────────────────────────────────────────────────────────────

DEMO_HOSPITALS = [
    Hospital(
        id="h-001",
        name="City General Hospital",
        total_beds=120,
        occupied_beds=87,
        available_beds=33,
    ),
    Hospital(
        id="h-002",
        name="Metro Medical Centre",
        total_beds=80,
        occupied_beds=52,
        available_beds=28,
    ),
    Hospital(
        id="h-003",
        name="Regional Trauma Centre",
        total_beds=200,
        occupied_beds=143,
        available_beds=57,
    ),
    Hospital(
        id="gov-central",
        name="Department of Health",
        total_beds=0,
        occupied_beds=0,
        available_beds=0,
    ),
]

DEMO_USERS = [
    User(
        id="u-nurse-001",
        name="Nurse Sarah",
        email="nurse@accidentcare.org",
        hashed_password=get_password_hash("demo"),
        role="nurse",
        hospital_id="h-001",
        hospital_name="City General Hospital",
    ),
    User(
        id="u-doctor-001",
        name="Dr. James Carter",
        email="doctor@accidentcare.org",
        hashed_password=get_password_hash("demo"),
        role="doctor",
        hospital_id="h-001",
        hospital_name="City General Hospital",
    ),
    User(
        id="u-admin-001",
        name="Admin Alice",
        email="admin@accidentcare.org",
        hashed_password=get_password_hash("demo"),
        role="admin",
        hospital_id="h-001",
        hospital_name="City General Hospital",
    ),
    User(
        id="u-gov-001",
        name="Gov. Official Robert",
        email="gov@accidentcare.org",
        hashed_password=get_password_hash("demo"),
        role="government",
        hospital_id="gov-central",
        hospital_name="Department of Health",
    ),
]

_today = date.today()

DEMO_PATIENTS = [
    Patient(
        id="p-001",
        name="Marcus Chen",
        age=34,
        gender="Male",
        admission_date=(_today - timedelta(days=2)).isoformat(),
        collision_type="Multi-Vehicle Pileup",
        injury_description="Severe skull fracture and internal hemorrhage from highway pileup.",
        severity="Critical",
        priority="High",
        predicted_los_days=15,
        created_by="u-nurse-001",
        hospital_id="h-001",
    ),
    Patient(
        id="p-002",
        name="Priya Sharma",
        age=28,
        gender="Female",
        admission_date=(_today - timedelta(days=1)).isoformat(),
        collision_type="Vehicle-Pedestrian",
        injury_description="Fractured pelvis and laceration on left arm.",
        severity="Moderate",
        priority="Medium",
        predicted_los_days=6,
        created_by="u-nurse-001",
        hospital_id="h-001",
    ),
    Patient(
        id="p-003",
        name="David Osei",
        age=45,
        gender="Male",
        admission_date=_today.isoformat(),
        collision_type="Vehicle-Vehicle",
        injury_description="Whiplash injury and minor concussion.",
        severity="Minor",
        priority="Low",
        predicted_los_days=2,
        created_by="u-nurse-001",
        hospital_id="h-001",
    ),
    Patient(
        id="p-004",
        name="Elena Vasquez",
        age=67,
        gender="Female",
        admission_date=(_today - timedelta(days=3)).isoformat(),
        collision_type="Single Vehicle",
        injury_description="Spinal trauma and rib fractures. Unconscious at scene.",
        severity="Critical",
        priority="High",
        predicted_los_days=18,
        created_by="u-nurse-001",
        hospital_id="h-001",
    ),
    Patient(
        id="p-005",
        name="Tom Baker",
        age=22,
        gender="Male",
        admission_date=(_today - timedelta(days=1)).isoformat(),
        collision_type="Vehicle-Bicycle",
        injury_description="Broken arm and severe road rash.",
        severity="Moderate",
        priority="Medium",
        predicted_los_days=4,
        created_by="u-nurse-001",
        hospital_id="h-001",
    ),
]

DEMO_TRANSFERS = [
    Transfer(
        id="t-001",
        patient_name="Marcus Chen",
        source_hospital="City General Hospital",
        destination_hospital="Regional Trauma Centre",
        status="pending",
        requested_date=(_today - timedelta(days=1)).isoformat(),
    ),
    Transfer(
        id="t-002",
        patient_name="Elena Vasquez",
        source_hospital="City General Hospital",
        destination_hospital="Metro Medical Centre",
        status="approved",
        requested_date=(_today - timedelta(days=3)).isoformat(),
    ),
    Transfer(
        id="t-003",
        patient_name="Jane Doe",
        source_hospital="Metro Medical Centre",
        destination_hospital="City General Hospital",
        status="rejected",
        requested_date=(_today - timedelta(days=5)).isoformat(),
    ),
]

DEMO_AUDIT_LOGS = [
    AuditLog(
        id=str(uuid.uuid4()),
        user="Nurse Sarah",
        role="nurse",
        action="Patient 'Marcus Chen' admitted. Severity: Critical, Priority: High, LOS: 15 days.",
        timestamp=(_today - timedelta(days=2)).isoformat() + "T08:14:22Z",
    ),
    AuditLog(
        id=str(uuid.uuid4()),
        user="Nurse Sarah",
        role="nurse",
        action="Patient 'Priya Sharma' admitted. Severity: Moderate, Priority: Medium, LOS: 6 days.",
        timestamp=(_today - timedelta(days=1)).isoformat() + "T09:32:11Z",
    ),
    AuditLog(
        id=str(uuid.uuid4()),
        user="Admin Alice",
        role="admin",
        action="Transfer for 'Elena Vasquez' (City General Hospital → Metro Medical Centre) marked as 'approved'.",
        timestamp=(_today - timedelta(days=3)).isoformat() + "T14:05:47Z",
    ),
    AuditLog(
        id=str(uuid.uuid4()),
        user="Admin Alice",
        role="admin",
        action="Transfer for 'Jane Doe' (Metro Medical Centre → City General Hospital) marked as 'rejected'.",
        timestamp=(_today - timedelta(days=5)).isoformat() + "T16:22:03Z",
    ),
    AuditLog(
        id=str(uuid.uuid4()),
        user="Nurse Sarah",
        role="nurse",
        action="Patient 'David Osei' admitted. Severity: Minor, Priority: Low, LOS: 2 days.",
        timestamp=_today.isoformat() + "T07:45:00Z",
    ),
]


# ── Init function ─────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables and insert seed data if the DB is empty."""
    Base.metadata.create_all(bind=engine)

    db: Session = SessionLocal()
    try:
        # Only seed if no users exist yet (idempotent startup)
        if db.query(User).count() > 0:
            return

        for hospital in DEMO_HOSPITALS:
            db.merge(hospital)  # upsert by primary key
        for user in DEMO_USERS:
            db.merge(user)
        for patient in DEMO_PATIENTS:
            db.merge(patient)
        for transfer in DEMO_TRANSFERS:
            db.merge(transfer)
        for log in DEMO_AUDIT_LOGS:
            db.add(log)

        db.commit()
        print("✅ Database initialised with demo seed data.")
    except Exception as exc:
        db.rollback()
        print(f"⚠️  Seed error: {exc}")
        raise
    finally:
        db.close()
