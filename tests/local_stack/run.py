"""Start the local pre-deployment stack and keep it running for browser tests.

    TEST_NURSE_EMAIL=... TEST_NURSE_PASSWORD=... (all eight TEST_* variables,
    or an untracked .env.test) python -m tests.local_stack.run [--port 8000]

Builds a throwaway PostgreSQL cluster from supabase/migrations (applied twice),
starts PostgREST in Docker, the in-memory LocalAuth and the real backend on
http://127.0.0.1:<port>. The frontend dev server proxies /api to port 8000, so
`npx playwright test` in the frontend then runs the live-backend specs with
the same TEST_* variables. Ctrl-C stops everything and deletes the cluster.
Nothing here connects to Supabase.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from demo_accounts import load_test_accounts
from tests.db.conftest import Cluster, _pg_bin, free_port
from tests.local_stack.stack import local_stack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000, help="port for the backend (default 8000)")
    args = parser.parse_args()
    accounts = {role: {"email": a["email"], "password": a["password"]} for role, a in load_test_accounts().items()}

    initdb, pg_ctl = _pg_bin("initdb"), _pg_bin("pg_ctl")
    if not (initdb and pg_ctl and _pg_bin("psql")):
        raise SystemExit("PostgreSQL server binaries (initdb/pg_ctl/psql) are required.")
    workdir = Path(tempfile.mkdtemp(prefix="accident-care-stack-"))
    data_dir, port = workdir / "data", free_port()
    subprocess.run([initdb, "-D", str(data_dir), "-U", "postgres", "--auth=trust", "-E", "UTF8"],
                   check=True, capture_output=True)
    subprocess.run([pg_ctl, "-D", str(data_dir), "-w", "-l", str(workdir / "server.log"), "-o",
                    f"-p {port} -c listen_addresses=127.0.0.1 -c unix_socket_directories=''", "start"],
                   check=True, capture_output=True)
    try:
        cluster = Cluster(data_dir, port)
        cluster.create_database("local_stack")
        cluster.migrate("local_stack")
        cluster.migrate("local_stack")          # the chain is re-runnable
        log = workdir / "backend.log"
        with local_stack(cluster, "local_stack", accounts, api_port=args.port, log_path=log) as stack:
            print(f"Backend:  {stack['api']}  (health: {stack['api']}/health, log: {log})")
            print(f"Staff accounts belong to {list(stack['hospitals'])[0]}. Ctrl-C to stop.")
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        subprocess.run([pg_ctl, "-D", str(data_dir), "-m", "fast", "stop"], capture_output=True)
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
