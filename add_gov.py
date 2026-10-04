#!/usr/bin/env python3
"""Script to add a new Government Officer to the Supabase project."""
import argparse
import getpass
import os
import sys

from dotenv import load_dotenv
from supabase import Client, create_client

def main():
    parser = argparse.ArgumentParser(description="Add a Government Officer to the Accident Care system.")
    parser.add_argument("--email", required=True, help="Email for the new officer")
    parser.add_argument("--name", required=True, help="Full name of the officer")
    
    args = parser.parse_args()

    # Prompted, not a command-line argument: arguments end up in shell history
    # and in the process list.
    password = getpass.getpass(f"Password for {args.email} (min 8 chars): ")
    if len(password) < 8 or password != getpass.getpass("Repeat the password: "):
        print("Error: the passwords are shorter than 8 characters or do not match.")
        sys.exit(1)

    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        print("Error: SUPABASE_URL or SUPABASE_KEY is missing in .env")
        sys.exit(1)

    supabase: Client = create_client(url, key)

    print(f"Checking if {args.email} already exists...")
    auth_users = supabase.auth.admin.list_users(per_page=1000)
    existing_user = next((u for u in auth_users if u.email == args.email), None)

    if existing_user:
        print(f"Auth user {args.email} already exists (ID: {existing_user.id}). Updating password...")
        supabase.auth.admin.update_user_by_id(existing_user.id, {
            "password": password,
            "email_confirm": True,
        })
        user_id = existing_user.id
    else:
        print(f"Creating new auth user for {args.email}...")
        response = supabase.auth.admin.create_user({
            "email": args.email,
            "password": password,
            "email_confirm": True,
        })
        if not response.user:
            print(f"Error: Authentication account was not created for {args.email}")
            sys.exit(1)
        user_id = response.user.id

    # Now add/update in the Government_Officer table
    print("Ensuring user record exists in the public.Government_Officer table...")
    profile = (
        supabase.table("Government_Officer")
        .select("officer_id")
        .eq("email", args.email)
        .limit(1)
        .execute()
    )

    if profile.data:
        print("Updating existing Government Officer profile...")
        supabase.table("Government_Officer").update({"name": args.name}).eq("email", args.email).execute()
    else:
        print("Inserting new Government Officer profile...")
        supabase.table("Government_Officer").insert({
            "officer_id": user_id,
            "email": args.email,
            "name": args.name,
        }).execute()

    print(f"\nSuccess! Government Officer {args.name} ({args.email}) is ready.")

if __name__ == "__main__":
    main()
