"""Supabase client singleton + FastAPI dependency."""
from supabase import create_client, Client

from app.core.config import get_settings

settings = get_settings()

# Supabase client (uses service role key for full DB access)
supabase: Client = create_client(settings.supabase_url, settings.supabase_key)


def get_supabase() -> Client:
    """Provide the service-role client used for database and admin operations."""
    return supabase


def get_auth_supabase() -> Client:
    """Provide an isolated client because password sign-in mutates client session state."""
    return create_client(settings.supabase_url, settings.supabase_key)
