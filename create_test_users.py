"""Idempotently provision demo logins in the configured Supabase project."""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if not url or not key:
    print("SUPABASE_URL or SUPABASE_KEY is missing in .env")
    sys.exit(1)

supabase: Client = create_client(url, key)
password = "DemoPass123!"

hospital_result = (
    supabase.table("Hospital")
    .select("hospital_id")
    .eq("name", "General Hospital Colombo")
    .limit(1)
    .execute()
)
if hospital_result.data:
    hospital_id = hospital_result.data[0]["hospital_id"]
else:
    hospital = supabase.table("Hospital").insert({
        "name": "General Hospital Colombo",
        "type": "Teaching",
        "region": "Western",
        "district": "Colombo",
        "address": "Regent St, Colombo 00800",
    }).execute()
    hospital_id = hospital.data[0]["hospital_id"]

accounts = [
    ("nurse@accidentcare.org", "Demo Nurse", "nurse"),
    ("doctor@accidentcare.org", "Demo Doctor", "doctor"),
    ("admin@accidentcare.org", "Demo Administrator", "admin"),
    ("gov@accidentcare.org", "Demo Government Officer", "government"),
]

auth_users = {user.email: user for user in supabase.auth.admin.list_users(per_page=1000)}

for email, name, role in accounts:
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

print("Demo accounts are ready. Password for all accounts: DemoPass123!")
for email, _, role in accounts:
    print(f"- {role}: {email}")
