-- Severity-model raw inputs that the API writes on intake. They were added to
-- the live project by hand (the old root supabase_model_integration.sql) but
-- never to supabase/migrations, so a fresh database could not accept an intake.
-- Idempotent: a no-op where the columns already exist.

alter table public."Accident_Record"
    add column if not exists mode_of_travel_during_accident text,
    add column if not exists passenger_type text,
    add column if not exists other_injury text;

notify pgrst, 'reload schema';
