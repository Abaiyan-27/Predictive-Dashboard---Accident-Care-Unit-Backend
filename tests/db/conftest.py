"""A throwaway PostgreSQL cluster built only from supabase/migrations.

Nothing here touches Supabase: each test session initialises a private cluster
(TCP on a free localhost port, no Unix socket), creates databases from the
repository migrations alone, and stops the cluster afterwards. The tests are
skipped when PostgreSQL server binaries are not installed.
"""
from __future__ import annotations

import contextlib
import glob
import shutil
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

psycopg = pytest.importorskip("psycopg")

BACKEND_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = BACKEND_ROOT / "supabase" / "migrations"
MIGRATIONS = sorted(MIGRATIONS_DIR.glob("*.sql"))


def _pg_bin(name: str) -> str | None:
    found = shutil.which(name)
    if found and name != "psql":
        return found
    candidates = sorted(glob.glob(f"/usr/lib/postgresql/*/bin/{name}"), reverse=True)
    return candidates[0] if candidates else found


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Cluster:
    def __init__(self, data_dir: Path, port: int):
        self.data_dir = data_dir
        self.port = port

    def dsn(self, dbname: str = "postgres") -> str:
        return f"host=127.0.0.1 port={self.port} user=postgres dbname={dbname}"

    def create_database(self, name: str) -> str:
        with psycopg.connect(self.dsn(), autocommit=True) as conn:
            conn.execute(f'drop database if exists "{name}"')
            conn.execute(f'create database "{name}"')
        return name

    def apply(self, dbname: str, migration: Path) -> subprocess.CompletedProcess:
        """Apply one migration the way a deploy would: one transaction, stop on error."""
        return subprocess.run(
            [_pg_bin("psql"), "-h", "127.0.0.1", "-p", str(self.port), "-U", "postgres",
             "-d", dbname, "-q", "-X", "-1", "-v", "ON_ERROR_STOP=1", "-f", str(migration)],
            capture_output=True, text=True,
        )

    def migrate(self, dbname: str, migrations=None) -> list[subprocess.CompletedProcess]:
        results = []
        for migration in migrations if migrations is not None else MIGRATIONS:
            result = self.apply(dbname, migration)
            assert result.returncode == 0, f"{migration.name} failed:\n{result.stderr}"
            results.append(result)
        return results


@pytest.fixture(scope="session")
def pg_cluster(tmp_path_factory):
    initdb, pg_ctl, psql = _pg_bin("initdb"), _pg_bin("pg_ctl"), _pg_bin("psql")
    if not (initdb and pg_ctl and psql):
        pytest.skip("PostgreSQL server binaries (initdb/pg_ctl/psql) are not installed")

    data_dir = tmp_path_factory.mktemp("pg") / "data"
    port = free_port()
    subprocess.run([initdb, "-D", str(data_dir), "-U", "postgres", "--auth=trust", "-E", "UTF8"],
                   check=True, capture_output=True)
    subprocess.run(
        [pg_ctl, "-D", str(data_dir), "-w", "-l", str(data_dir.parent / "server.log"),
         "-o", f"-p {port} -c listen_addresses=127.0.0.1 -c unix_socket_directories=''",
         "start"],
        check=True, capture_output=True,
    )
    cluster = Cluster(data_dir, port)
    try:
        yield cluster
    finally:
        subprocess.run([pg_ctl, "-D", str(data_dir), "-m", "fast", "stop"], capture_output=True)


@pytest.fixture(scope="session")
def migrated_db(pg_cluster) -> str:
    """A fresh database built from every migration, applied twice to prove idempotency."""
    name = pg_cluster.create_database("migrated")
    pg_cluster.migrate(name)
    pg_cluster.migrate(name)
    return name


@pytest.fixture
def conn(pg_cluster, migrated_db):
    """A connection whose work is rolled back after each test."""
    with psycopg.connect(pg_cluster.dsn(migrated_db)) as connection:
        yield connection
        connection.rollback()


# ── The real API on top of a migrated database ─────────────────────────────
# A PostgREST container serves the database and a small proxy adds the
# /rest/v1 prefix the Supabase client expects. Skipped without Docker.

POSTGREST_IMAGE = "postgrest/postgrest:v12.2.3"
JWT_SECRET = "test-postgrest-secret-that-is-at-least-32-chars"

# Supabase provides these roles; a bare PostgreSQL cluster needs them created.
SUPABASE_ROLES = """
do $$ begin
    if not exists (select from pg_roles where rolname = 'anon') then create role anon nologin; end if;
    if not exists (select from pg_roles where rolname = 'service_role') then
        create role service_role nologin bypassrls; end if;
    if not exists (select from pg_roles where rolname = 'authenticator') then
        create role authenticator login noinherit; end if;
end $$;
grant anon, service_role to authenticator;
grant usage on schema public to anon, service_role;
grant all on all tables in schema public to service_role;
grant all on all sequences in schema public to service_role;
grant execute on all functions in schema public to service_role;
"""


def docker_ready() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", POSTGREST_IMAGE],
                          capture_output=True).returncode == 0


def _proxy_handler(upstream: str):
    class Proxy(BaseHTTPRequestHandler):
        def _forward(self):
            path = self.path.removeprefix("/rest/v1")
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            headers = {k: v for k, v in self.headers.items() if k.lower() not in {"host", "content-length"}}
            response = httpx.request(self.command, upstream + path, headers=headers,
                                     content=body, timeout=30)
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                if key.lower() not in {"transfer-encoding", "connection", "content-encoding", "content-length"}:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(response.content)))
            self.end_headers()
            self.wfile.write(response.content)

        do_GET = do_POST = do_PATCH = do_DELETE = do_PUT = do_HEAD = _forward

        def log_message(self, *args):
            pass

    return Proxy


@contextlib.contextmanager
def running_api(cluster: Cluster, dbname: str):
    """Yield (TestClient, autocommit connection) for the real app on `dbname`."""
    if not docker_ready():
        pytest.skip(f"Docker with {POSTGREST_IMAGE} is not available")

    from fastapi.testclient import TestClient
    from jose import jwt
    from supabase import create_client

    from app.db.session import get_supabase
    from app.main import app

    with psycopg.connect(cluster.dsn(dbname), autocommit=True) as setup:
        setup.execute(SUPABASE_ROLES)

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

    upstream = f"http://127.0.0.1:{rest_port}"
    server = ThreadingHTTPServer(("127.0.0.1", free_port()), _proxy_handler(upstream))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        for _ in range(60):
            try:
                if httpx.get(upstream + "/", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        else:
            pytest.fail("PostgREST did not start")

        service_key = jwt.encode({"role": "service_role"}, JWT_SECRET, algorithm="HS256")
        client = create_client(f"http://127.0.0.1:{server.server_port}", service_key)
        app.dependency_overrides[get_supabase] = lambda: client
        with TestClient(app) as test_client, psycopg.connect(cluster.dsn(dbname), autocommit=True) as conn:
            yield test_client, conn
    finally:
        app.dependency_overrides.clear()
        server.shutdown()
        subprocess.run(["docker", "stop", container], capture_output=True)
