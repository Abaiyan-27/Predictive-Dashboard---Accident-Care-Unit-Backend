-- Run once in the Supabase SQL editor before using the expanded intake and
-- treatment forms. Statements are idempotent.

-- Severity model input (converted to the notebook's collision-time bucket).
alter table public."Accident_Record"
    add column if not exists incident_at timestamptz,

    -- Transfer and/or discharge model inputs.
    add column if not exists road_condition text,
    add column if not exists visibility text,
    add column if not exists road_signals_exist text,
    add column if not exists collision_force_from text,
    add column if not exists alcohol_consumption text,
    add column if not exists illicit_drugs text,

    -- Severity model raw inputs.
    add column if not exists mode_of_travel_during_accident text,
    add column if not exists passenger_type text,
    add column if not exists other_injury text;

-- Stay-duration, transfer, and discharge model inputs.
alter table public."Treatment"
    add column if not exists investigation_done text,
    add column if not exists management_name_1 text,
    add column if not exists management_name_2 text;

-- Discharge model input.
alter table public."Transfer"
    add column if not exists state_of_transfer text;

notify pgrst, 'reload schema';
