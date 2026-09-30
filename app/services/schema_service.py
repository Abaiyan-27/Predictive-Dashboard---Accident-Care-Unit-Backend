"""Checks for the Supabase columns required by the model-integrated API."""
from collections.abc import Iterable
from typing import Any

from postgrest.exceptions import APIError
from supabase import Client


MIGRATION_FILE = "supabase_model_integration.sql"

MODEL_INTEGRATION_SCHEMA: dict[str, tuple[str, ...]] = {
    "Accident_Record": (
        "incident_at",
        "road_condition",
        "visibility",
        "road_signals_exist",
        "collision_force_from",
        "alcohol_consumption",
        "illicit_drugs",
        "mode_of_travel_during_accident",
        "passenger_type",
        "other_injury",
    ),
    "Treatment": (
        "investigation_done",
        "management_name_1",
        "management_name_2",
    ),
    "Transfer": ("state_of_transfer",),
}

# PostgREST/Postgres codes for a missing table or column.
_MISSING_SCHEMA_CODES = {"PGRST204", "PGRST205", "42P01", "42703"}


class DatabaseMigrationRequired(RuntimeError):
    """Raised when the live Supabase schema is older than the API schema."""

    def __init__(self, table: str, message: str | None = None) -> None:
        self.table = table
        self.database_message = message
        detail = (
            "Database setup is incomplete for the model-integrated forms. "
            f"Run {MIGRATION_FILE} in the Supabase SQL Editor, then retry. "
            f"Missing schema was detected for {table}."
        )
        if message:
            detail = f"{detail} Supabase: {message}"
        super().__init__(detail)


def require_table_columns(
    supabase: Client,
    table: str,
    columns: Iterable[str],
) -> None:
    """Verify a table/column set without reading any patient values."""
    try:
        supabase.table(table).select(",".join(columns)).limit(1).execute()
    except APIError as exc:
        if exc.code in _MISSING_SCHEMA_CODES:
            raise DatabaseMigrationRequired(table, exc.message) from exc
        raise


def require_intake_schema(supabase: Client) -> None:
    """Verify everything required before creating a new patient intake."""
    require_table_columns(
        supabase,
        "Accident_Record",
        MODEL_INTEGRATION_SCHEMA["Accident_Record"],
    )


def get_model_integration_schema_status(supabase: Client) -> dict[str, Any]:
    """Return a non-throwing readiness report for every migrated table."""
    tables: dict[str, dict[str, Any]] = {}
    for table, columns in MODEL_INTEGRATION_SCHEMA.items():
        try:
            require_table_columns(supabase, table, columns)
            tables[table] = {"ready": True, "message": None}
        except DatabaseMigrationRequired as exc:
            tables[table] = {
                "ready": False,
                "message": exc.database_message or str(exc),
            }
        except Exception as exc:
            tables[table] = {
                "ready": False,
                "message": f"Schema check failed: {exc}",
            }

    return {
        "ready": all(item["ready"] for item in tables.values()),
        "migration": MIGRATION_FILE,
        "tables": tables,
    }
