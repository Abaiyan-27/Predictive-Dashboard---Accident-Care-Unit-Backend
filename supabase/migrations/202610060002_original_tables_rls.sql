-- Close the original tables to Supabase's public keys.
--
-- No repository migration enabled row level security on the eleven tables of
-- the base schema. On Supabase the anon and authenticated roles receive table
-- privileges by default, so without RLS anyone holding the project's public
-- (anon) key could read and write patient records through PostgREST directly,
-- bypassing this API's authorization.
--
-- The API connects with the service-role key, which bypasses RLS, so enabling
-- RLS with no policies changes nothing for the application and denies direct
-- anon/authenticated access. Same approach as the prediction tables
-- (202610030004). Idempotent; no rows are read or changed.
--
-- Production status was not verified from this repository: apply this
-- migration (or confirm RLS in the Supabase dashboard) before relying on it.

alter table public."Hospital"           enable row level security;
alter table public."User"               enable row level security;
alter table public."Government_Officer" enable row level security;
alter table public."Patient"            enable row level security;
alter table public."Socioeconomic_Info" enable row level security;
alter table public."Accident_Record"    enable row level security;
alter table public."Injury_Description" enable row level security;
alter table public."Treatment"          enable row level security;
alter table public."Transfer"           enable row level security;
alter table public."Hospital_Patient"   enable row level security;
alter table public."AuditLog"           enable row level security;

notify pgrst, 'reload schema';
