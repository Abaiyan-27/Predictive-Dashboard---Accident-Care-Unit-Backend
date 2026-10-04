import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from datetime import datetime, timedelta, timezone

import os
from jose import jwt as _jwt
os.environ["SECRET_KEY"] = "test-secret-key-that-is-at-least-32-chars-long"
os.environ["ALGORITHM"] = "HS256"
os.environ["SUPABASE_URL"] = "http://localhost:8000"
# A JWT-shaped placeholder (the Supabase client checks the shape); never a real key.
os.environ["SUPABASE_KEY"] = _jwt.encode({"role": "service_role"}, "unit-tests-only")

from app.main import app
from app.db.session import get_supabase, get_auth_supabase
from app.core.security import create_access_token

@pytest.fixture
def mock_supabase():
    mock_client = MagicMock()
    return mock_client

@pytest.fixture
def mock_auth_supabase():
    mock_client = MagicMock()
    return mock_client

@pytest.fixture
def client(mock_supabase, mock_auth_supabase):
    app.dependency_overrides[get_supabase] = lambda: mock_supabase
    app.dependency_overrides[get_auth_supabase] = lambda: mock_auth_supabase
    
    with TestClient(app) as test_client:
        yield test_client
        
    app.dependency_overrides.clear()

def generate_test_token(user_id: str, role: str) -> str:
    """Helper to generate valid JWTs for testing."""
    return create_access_token(
        data={"sub": user_id, "role": role},
        expires_delta=timedelta(minutes=30)
    )

@pytest.fixture
def nurse_token():
    return generate_test_token("nurse-id-123", "nurse")

@pytest.fixture
def doctor_token():
    return generate_test_token("doctor-id-123", "doctor")

@pytest.fixture
def admin_token():
    return generate_test_token("admin-id-123", "admin")

@pytest.fixture
def government_token():
    return generate_test_token("gov-id-123", "government")

@pytest.fixture
def mock_intake_payload():
    return {
        "full_name": "John Doe",
        "age": "35",
        "gender": "male",
        "hospital_id": "00000000-0000-0000-0000-000000000002",
        "first_hospital_id": "00000000-0000-0000-0000-000000000002",
        "vehicle_type": "car",
        "helmet_worn": "no",
        "road_type": "highway",
        "speed": "60-80",
        "incident_at": "2024-01-01T12:00:00Z",
        "road_condition": "dry",
        "visibility": "good",
        "road_signals_exist": "yes",
        "collision_force_from": "front",
        "alcohol_consumption": "no",
        "illicit_drugs": "no",
        "mode_of_transport": "ambulance",
        "time_to_hospital": "30_mins",
        "collision_with": "another_vehicle",
        "injuries": [
            {
                "site_of_injury": "head",
                "type_of_injury": "concussion"
            }
        ]
    }
