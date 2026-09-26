import random
import json
import os
from locust import HttpUser, task, between, events

# Load pre-fetched tokens to avoid Supabase auth rate-limits during scale-up
try:
    with open("tokens.json", "r") as f:
        AUTH_TOKENS = json.load(f)
except FileNotFoundError:
    print("ERROR: tokens.json not found! Please run 'python fetch_tokens.py' first.")
    AUTH_TOKENS = {}

class BaseHospitalUser(HttpUser):
    abstract = True
    
    def on_start(self):
        """Use pre-fetched JWTs."""
        auth_data = AUTH_TOKENS.get(self.email)
        if not auth_data:
            print(f"No pre-fetched token found for {self.email}. Exiting.")
            self.environment.runner.quit()
            return
            
        self.token = auth_data.get("token")
        self.user_data = auth_data.get("user")
        self.hospital_id = self.user_data.get("hospital_id")
        
        # Set the Authorization header for all subsequent requests
        self.client.headers.update({
            "Authorization": f"Bearer {self.token}"
        })


class NurseUser(BaseHospitalUser):
    """
    Simulates a Nurse performing patient intake operations.
    High volume.
    """
    weight = 5
    wait_time = between(1, 5)
    email = "nurse@accidentcare.org"
    password = "DemoPass123!"

    @task(3)
    def submit_intake(self):
        """Simulate creating a new patient intake and running the ML severity model."""
        from datetime import datetime, timezone
        payload = {
            "full_name": f"Test Patient {random.randint(1000, 9999)}",
            "age": str(random.randint(18, 90)),
            "gender": random.choice(["Male", "Female", "Other"]),
            "hospital_id": self.hospital_id,
            "vehicle_type": random.choice(["Car", "Motorcycle", "Bicycle", "Pedestrian"]),
            "helmet_worn": random.choice(["Yes", "No", None]),
            "road_type": "Highway",
            "speed": str(random.randint(30, 120)),
            "incident_at": datetime.now(timezone.utc).isoformat(),
            "road_condition": "Dry",
            "visibility": "Clear",
            "road_signals_exist": random.choice(["Yes", "No", None]),
            "collision_force_from": "Front",
            "alcohol_consumption": random.choice(["Yes", "No"]),
            "illicit_drugs": random.choice(["Yes", "No", None]),
            "mode_of_transport": "Ambulance",
            "time_to_hospital": str(random.randint(10, 120)),
            "collision_with": "Vehicle",
            "injuries": [
                {
                    "site_of_injury": "Head",
                    "type_of_injury": "Laceration"
                }
            ]
        }
        self.client.post("/api/patients/intake", json=payload, name="/api/patients/intake")

    @task(1)
    def get_my_patients(self):
        """Simulate fetching the nurse's patient list."""
        self.client.get("/api/patients/my", name="/api/patients/my")

class DoctorUser(BaseHospitalUser):
    """
    Simulates a Doctor viewing patients and running the prognosis ML model.
    Medium volume.
    """
    weight = 3
    wait_time = between(2, 6)
    email = "doctor@accidentcare.org"
    password = "DemoPass123!"

    @task(2)
    def view_hospital_patients(self):
        """Simulate fetching all patients for the doctor's hospital."""
        if self.hospital_id:
            response = self.client.get(f"/api/patients/hospital/{self.hospital_id}", name="/api/patients/hospital/[id]")
            if response.status_code == 200:
                patients = response.json()
                if patients:
                    # Pick a random patient to view prognosis for
                    patient = random.choice(patients)
                    patient_id = patient.get("patient_id")
                    if patient_id:
                        self.client.get(f"/api/patients/{patient_id}/prognosis", name="/api/patients/[id]/prognosis")

class GovernmentUser(BaseHospitalUser):
    """
    Simulates a Government official fetching aggregate data and ML forecasts.
    Low volume.
    """
    weight = 1
    wait_time = between(5, 15)
    email = "gov@accidentcare.org"
    password = "DemoPass123!"

    @task
    def view_analytics(self):
        """Simulate fetching large analytical datasets."""
        self.client.get("/api/government/accident-trends", name="/api/government/accident-trends")
        self.client.get("/api/government/statistics", name="/api/government/statistics")
        self.client.get("/api/government/accident-forecast?horizon=6", name="/api/government/accident-forecast")
