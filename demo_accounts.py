"""Test/demo account credentials, read from the environment only.

Used by create_test_users.py and fetch_tokens.py (not by the API). Nothing here has a default:
set the variables (see .env.test.example) in your shell or in a local,
untracked .env.test file. Never point these scripts at production.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROLES = ("nurse", "doctor", "admin", "government")
NAMES = {
    "nurse": "Demo Nurse",
    "doctor": "Demo Doctor",
    "admin": "Demo Administrator",
    "government": "Demo Government Officer",
}


def _load_env_test() -> None:
    """Read KEY=VALUE lines from .env.test (if present) without overriding the shell."""
    path = Path(__file__).resolve().parent / ".env.test"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_test_accounts(roles: tuple[str, ...] = ROLES) -> dict[str, dict[str, str]]:
    """{role: {"email", "password", "name"}}; exits with an explanation if any is unset."""
    _load_env_test()
    accounts, missing = {}, []
    for role in roles:
        prefix = f"TEST_{role.upper()}"
        email, password = os.environ.get(f"{prefix}_EMAIL"), os.environ.get(f"{prefix}_PASSWORD")
        if not email:
            missing.append(f"{prefix}_EMAIL")
        if not password:
            missing.append(f"{prefix}_PASSWORD")
        accounts[role] = {"email": email or "", "password": password or "", "name": NAMES[role]}
    if missing:
        sys.exit(
            "Test account credentials are not configured: " + ", ".join(missing)
            + ".\nSet them in the environment or in an untracked .env.test (see .env.test.example)."
        )
    return accounts
