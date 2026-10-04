"""A complete LOCAL stack for pre-deployment checks. Never touches Supabase.

    PostgreSQL (built only from supabase/migrations, applied twice)
      └─ PostgREST (Docker)            ← the real Supabase REST layer
    local proxy  /rest/v1 → PostgREST,  /auth/v1 → LocalAuth (in memory)
      └─ the real backend, `uvicorn app.main:app`, in its own process

LocalAuth is a minimal stand-in for Supabase Auth (GoTrue) covering the calls
this backend makes: password sign-in and the admin user API. It exists only
here, for local testing; the application code is unchanged and in deployment
talks to the real Supabase Auth.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from jose import jwt

from tests.db.conftest import JWT_SECRET, POSTGREST_IMAGE, SUPABASE_ROLES, Cluster, docker_ready, free_port, psycopg

BACKEND_ROOT = Path(__file__).resolve().parents[2]
HOSPITAL_A, HOSPITAL_B = "DGH – Vavuniya", "DGH – Kilinochchi"
STAFF_ROLES = ("nurse", "doctor", "admin")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalAuth:
    """In-memory Supabase Auth subset. Passwords are kept only as SHA-256 digests."""

    def __init__(self, service_key: str):
        self.service_key = service_key
        self.users: dict[str, dict] = {}
        self.lock = threading.Lock()

    @staticmethod
    def _digest(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def add_user(self, email: str, password: str, user_id: str | None = None, metadata: dict | None = None) -> dict:
        user = {
            "id": user_id or str(uuid.uuid4()), "aud": "authenticated", "role": "authenticated",
            "email": email.lower(), "app_metadata": {"provider": "email"},
            "user_metadata": metadata or {}, "created_at": _now(), "email_confirmed_at": _now(),
            "last_sign_in_at": None, "_password": self._digest(password),
        }
        with self.lock:
            self.users[user["id"]] = user
        return user

    @staticmethod
    def public(user: dict) -> dict:
        return {k: v for k, v in user.items() if not k.startswith("_")}

    def handle(self, method: str, path: str, query: dict, body: dict, headers: dict) -> tuple[int, dict]:
        if path == "/token" and method == "POST" and query.get("grant_type") == ["password"]:
            return self._sign_in(body)
        if path.startswith("/admin/"):
            if headers.get("authorization") != f"Bearer {self.service_key}":
                return 401, {"msg": "This endpoint requires a valid service-role key"}
            return self._admin(method, path.removeprefix("/admin"), query, body)
        return 404, {"msg": f"LocalAuth does not implement {method} {path}"}

    def _sign_in(self, body: dict) -> tuple[int, dict]:
        email = (body.get("email") or "").lower()
        with self.lock:
            user = next((u for u in self.users.values() if u["email"] == email), None)
            if not user or user["_password"] != self._digest(body.get("password") or ""):
                return 400, {"error": "invalid_grant", "error_description": "Invalid login credentials"}
            user["last_sign_in_at"] = _now()
        return 200, {"access_token": secrets.token_urlsafe(24), "refresh_token": secrets.token_urlsafe(24),
                     "token_type": "bearer", "expires_in": 3600, "expires_at": int(time.time()) + 3600,
                     "user": self.public(user)}

    def _admin(self, method: str, path: str, query: dict, body: dict) -> tuple[int, dict]:
        if path == "/users" and method == "GET":
            page, per_page = int(query.get("page", ["1"])[0] or 1), int(query.get("per_page", ["50"])[0] or 50)
            with self.lock:
                users = list(self.users.values())[(page - 1) * per_page: page * per_page]
            return 200, {"users": [self.public(u) for u in users], "aud": "authenticated"}
        if path == "/users" and method == "POST":
            email = (body.get("email") or "").lower()
            if any(u["email"] == email for u in self.users.values()):
                return 422, {"code": "email_exists", "msg": "A user with this email address has already been registered"}
            return 200, self.public(self.add_user(email, body.get("password") or "", metadata=body.get("user_metadata")))
        user_id = path.removeprefix("/users/")
        with self.lock:
            user = self.users.get(user_id)
            if not user:
                return 404, {"msg": "User not found"}
            if method == "PUT":
                if "password" in body:
                    user["_password"] = self._digest(body["password"])
                if "user_metadata" in body:
                    user["user_metadata"] = body["user_metadata"] or {}
                return 200, self.public(user)
            if method == "DELETE":
                del self.users[user_id]
                return 200, {}
        return 405, {"msg": "Method not allowed"}


def _proxy(rest_upstream: str, auth: LocalAuth):
    class Proxy(BaseHTTPRequestHandler):
        def _send(self, status: int, content: bytes, headers: dict | None = None):
            self.send_response(status)
            for key, value in (headers or {"Content-Type": "application/json"}).items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _handle(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            url = urlparse(self.path)
            if url.path.startswith("/auth/v1"):
                status, payload = auth.handle(
                    self.command, url.path.removeprefix("/auth/v1"), parse_qs(url.query),
                    json.loads(body or b"{}"), {k.lower(): v for k, v in self.headers.items()})
                return self._send(status, json.dumps(payload).encode())
            path = self.path.removeprefix("/rest/v1")
            headers = {k: v for k, v in self.headers.items() if k.lower() not in {"host", "content-length"}}
            response = httpx.request(self.command, rest_upstream + path, headers=headers, content=body, timeout=60)
            self._send(response.status_code, response.content, {
                k: v for k, v in response.headers.items()
                if k.lower() not in {"transfer-encoding", "connection", "content-encoding", "content-length"}})

        do_GET = do_POST = do_PATCH = do_DELETE = do_PUT = do_HEAD = _handle

        def log_message(self, *args):
            pass

    return Proxy


def _wait_for(url: str, seconds: float, what: str, process: subprocess.Popen | None = None):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"{what} exited with code {process.returncode}")
        try:
            if httpx.get(url, timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"{what} did not become ready at {url}")


@contextlib.contextmanager
def local_stack(cluster: Cluster, dbname: str, accounts: dict[str, dict], *, api_port: int | None = None,
                secret_key: str | None = None, log_path: Path | None = None):
    """Run the stack on an already-migrated database; yield connection details.

    ``accounts`` maps role → {"email", "password"} for nurse, doctor, admin and
    government; staff join HOSPITAL_A. The backend gets a SECRET_KEY generated
    here (or the one passed in) and reads no .env file.
    """
    if not docker_ready():
        raise RuntimeError(f"Docker with the {POSTGREST_IMAGE} image is required")
    with psycopg.connect(cluster.dsn(dbname), autocommit=True) as setup:
        setup.execute(SUPABASE_ROLES)
        setup.execute(f'alter database "{dbname}" set timezone to \'UTC\'')

    rest_port = free_port()
    container = subprocess.run(
        ["docker", "run", "-d", "--rm", "--network", "host",
         "-e", f"PGRST_DB_URI=postgres://authenticator@127.0.0.1:{cluster.port}/{dbname}",
         "-e", "PGRST_DB_SCHEMAS=public", "-e", "PGRST_DB_ANON_ROLE=anon",
         "-e", f"PGRST_JWT_SECRET={JWT_SECRET}",
         "-e", "PGRST_SERVER_HOST=127.0.0.1", "-e", f"PGRST_SERVER_PORT={rest_port}",
         POSTGREST_IMAGE],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    service_key = jwt.encode({"role": "service_role"}, JWT_SECRET, algorithm="HS256")
    auth = LocalAuth(service_key)
    proxy = ThreadingHTTPServer(("127.0.0.1", free_port()), _proxy(f"http://127.0.0.1:{rest_port}", auth))
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    backend = None
    log_file = None
    try:
        _wait_for(f"http://127.0.0.1:{rest_port}/", 30, "PostgREST")
        conn = psycopg.connect(cluster.dsn(dbname), autocommit=True)
        hospitals = {name: str(conn.execute('select hospital_id from "Hospital" where name = %s', (name,)).fetchone()[0])
                     for name in (HOSPITAL_A, HOSPITAL_B)}
        people = {}
        for role, account in accounts.items():
            user = auth.add_user(account["email"], account["password"])
            if role == "government":
                conn.execute('insert into "Government_Officer" (officer_id, name, email) values (%s, %s, %s)',
                             (user["id"], "Local Officer", account["email"]))
            else:
                conn.execute('insert into "User" (user_id, name, email, role, hospital_id) values (%s, %s, %s, %s, %s)',
                             (user["id"], f"Local {role.title()}", account["email"], role, hospitals[HOSPITAL_A]))
            people[role] = {"id": user["id"], **account}

        api_port = api_port or free_port()
        secret_key = secret_key or secrets.token_urlsafe(48)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("HF_", "SUPABASE_", "SECRET_KEY", "CORS_"))}
        env.update({
            "APP_ENV": "development", "SECRET_KEY": secret_key, "LOG_LEVEL": "INFO",
            "SUPABASE_URL": f"http://127.0.0.1:{proxy.server_port}", "SUPABASE_KEY": service_key,
            "CORS_ORIGINS": "http://localhost:5173", "PYTHONPATH": str(BACKEND_ROOT),
        })
        log_file = open(log_path, "w") if log_path else subprocess.DEVNULL
        # cwd outside the repository: the backend must not pick up a developer .env.
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(api_port)],
            cwd=tempfile.gettempdir(), env=env, stdout=log_file, stderr=subprocess.STDOUT,
        )
        api = f"http://127.0.0.1:{api_port}"
        _wait_for(f"{api}/health", 120, "The backend", backend)
        yield {"api": api, "db": conn, "auth": auth, "hospitals": hospitals, "people": people,
               "secret_key": secret_key, "service_key": service_key, "log_path": log_path}
        conn.close()
    finally:
        if backend is not None:
            backend.terminate()
            try:
                backend.wait(timeout=15)
            except subprocess.TimeoutExpired:
                backend.kill()
        if log_file not in (None, subprocess.DEVNULL):
            log_file.close()
        proxy.shutdown()
        subprocess.run(["docker", "stop", container], capture_output=True)
