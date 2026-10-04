"""Idempotently provision test logins in a DEVELOPMENT Supabase project.

Credentials come from TEST_*_EMAIL / TEST_*_PASSWORD (see demo_accounts.py and
.env.test.example); nothing is hardcoded. Existing Auth accounts with these
emails get their password reset, so the script refuses to run unless
--confirm-host names the Supabase host it will change. Never run it against
production.

    python create_test_users.py --confirm-host <project-ref>.supabase.co
"""
from __future__ import annotations

import argparse
import os
import sys
from urllib.parse import urlparse

from dotenv import load_dotenv
from supabase import Client, create_client

from demo_accounts import load_test_accounts

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--confirm-host", required=True,
                    help="The host of SUPABASE_URL, typed out to confirm the target project")
args = parser.parse_args()

load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if not url or not key:
    print("SUPABASE_URL or SUPABASE_KEY is missing in .env")
    sys.exit(1)
if urlparse(url).hostname != args.confirm_host.strip():
    sys.exit("--confirm-host does not match the host of SUPABASE_URL; nothing was changed.")

credentials = load_test_accounts()
supabase: Client = create_client(url, key)

# Staff test accounts join an existing hospital, named explicitly. The script
# never creates hospitals (the network hospitals come from the migrations).
hospital_name = os.environ.get("TEST_HOSPITAL_NAME")
if not hospital_name:
    sys.exit("TEST_HOSPITAL_NAME is not set: name the existing hospital the staff test accounts belong to.")
hospital_result = (
    supabase.table("Hospital")
    .select("hospital_id")
    .eq("name", hospital_name)
    .limit(1)
    .execute()
)
if not hospital_result.data:
    sys.exit(f"No hospital named {hospital_name!r}; nothing was changed.")
hospital_id = hospital_result.data[0]["hospital_id"]

accounts = [(item["email"], item["name"], role, item["password"]) for role, item in credentials.items()]

auth_users = {user.email: user for user in supabase.auth.admin.list_users(per_page=1000)}

for email, name, role, password in accounts:
    auth_user = auth_users.get(email)
    if auth_user:
        supabase.auth.admin.update_user_by_id(auth_user.id, {
            "password": password,
            "email_confirm": True,
        })
    else:
        response = supabase.auth.admin.create_user({
            "email": email,
            "password": password,
            "email_confirm": True,
        })
        if not response.user:
            raise RuntimeError(f"Authentication account was not created for {email}")
        auth_user = response.user

    if role == "government":
        profile = (
            supabase.table("Government_Officer")
            .select("officer_id")
            .eq("email", email)
            .limit(1)
            .execute()
        )
        if profile.data:
            supabase.table("Government_Officer").update({"name": name}).eq(
                "email", email
            ).execute()
        else:
            supabase.table("Government_Officer").insert({
                "officer_id": auth_user.id,
                "name": name,
                "email": email,
            }).execute()
    else:
        profile = (
            supabase.table("User")
            .select("user_id")
            .eq("email", email)
            .limit(1)
            .execute()
        )
        values = {
            "name": name,
            "role": role,
            "hospital_id": hospital_id,
        }
        if profile.data:
            supabase.table("User").update(values).eq("email", email).execute()
        else:
            supabase.table("User").insert({
                "user_id": auth_user.id,
                "email": email,
                **values,
            }).execute()

print("Test accounts are ready (passwords are the TEST_*_PASSWORD values; not printed).")
for email, _, role, _ in accounts:
    print(f"- {role}: {email}")
