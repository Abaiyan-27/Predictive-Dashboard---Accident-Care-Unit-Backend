import json
import os
import httpx

# The URLs might be different based on configuration, but we'll use the local backend
API_BASE = "http://localhost:8000/api"

users = [
    {"email": "nurse@accidentcare.org", "password": "DemoPass123!"},
    {"email": "doctor@accidentcare.org", "password": "DemoPass123!"},
    {"email": "admin@accidentcare.org", "password": "DemoPass123!"},
    {"email": "gov@accidentcare.org", "password": "DemoPass123!"}
]

tokens = {}

print("Fetching auth tokens to avoid Supabase rate-limits during load test...")
with httpx.Client(timeout=30.0) as client:
    for user in users:
        print(f"Logging in {user['email']}...")
        response = client.post(f"{API_BASE}/auth/login", json=user)
        if response.status_code == 200:
            data = response.json()
            tokens[user['email']] = {
                "token": data["token"],
                "user": data["user"]
            }
            print(f"Success for {user['email']}")
        else:
            print(f"Failed to log in {user['email']}: {response.text}")

with open("tokens.json", "w") as f:
    json.dump(tokens, f, indent=2)

print("Tokens saved to tokens.json")
