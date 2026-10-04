"""Log the test accounts in once and save their API session tokens for locustfile.py.

Credentials come from TEST_*_EMAIL / TEST_*_PASSWORD (see demo_accounts.py);
the API defaults to the local backend. The output file holds live session
tokens: it is git-ignored and must never be committed.
"""
import json
import os

import httpx

from demo_accounts import load_test_accounts

API_BASE = os.environ.get("LOAD_TEST_API_BASE", "http://localhost:8000/api")
TOKENS_FILE = os.environ.get("LOAD_TEST_TOKENS_FILE", "tokens.json")

tokens = {}

print(f"Fetching session tokens from {API_BASE} to avoid Supabase rate limits during the load test...")
with httpx.Client(timeout=30.0) as client:
    for role, account in load_test_accounts().items():
        print(f"Logging in the {role} test account...")
        response = client.post(f"{API_BASE}/auth/login",
                               json={"email": account["email"], "password": account["password"]})
        if response.status_code == 200:
            data = response.json()
            tokens[role] = {"token": data["token"], "user": data["user"]}
            print(f"Success for the {role} account")
        else:
            print(f"Failed to log in the {role} account: HTTP {response.status_code}")

with open(TOKENS_FILE, "w") as f:
    json.dump(tokens, f, indent=2)
os.chmod(TOKENS_FILE, 0o600)

print(f"Tokens saved to {TOKENS_FILE} (git-ignored; delete it after the load test)")
