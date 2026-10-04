-- Base schema: the tables the API was built on, as they existed before
-- 202608190001_model_integration.sql. Reconstructed from the live Supabase
-- project (column names, types, defaults, nullability and foreign keys).
--
-- Every statement is CREATE ... IF NOT EXISTS, so this migration is a no-op on
-- the existing project and only builds the schema on a fresh database.
-- Later migrations add columns and tighten constraints.

create table if not exists public."Hospital" (
    hospital_id uuid primary key default gen_random_uuid(),
    name        text not null,
    type        text,
    region      text,
    district    text,
    address     text
);

-- Staff accounts (nurse, doctor, admin). user_id matches the Supabase Auth user id.
create table if not exists public."User" (
    user_id     uuid primary key default gen_random_uuid(),
    name        text not null,
    email       text not null,
    nic         text,
    role        text not null,
    hospital_id uuid references public."Hospital" (hospital_id),
    created_at  timestamptz default now()
);

-- Government officers are deliberately separate and have no hospital.
create table if not exists public."Government_Officer" (
    officer_id uuid primary key default gen_random_uuid(),
    name       text not null,
    email      text not null,
    nic        text,
    created_at timestamptz default now()
);

create table if not exists public."Patient" (
    patient_id uuid primary key default gen_random_uuid(),
    full_name  text,
    age        text,
    gender     text,
    ethnicity  text,
    occupation text
);

create table if not exists public."Socioeconomic_Info" (
    patient_id            uuid primary key references public."Patient" (patient_id),
    family_income_before  text,
    family_income_after   text,
    family_current_status text,
    insurance_claim_type  text
);

create table if not exists public."Accident_Record" (
    accident_id       uuid primary key default gen_random_uuid(),
    patient_id        uuid references public."Patient" (patient_id),
    hospital_id       uuid references public."Hospital" (hospital_id),
    first_hospital_id uuid references public."Hospital" (hospital_id),
    managed_by        uuid references public."User" (user_id),
    created_by        uuid references public."User" (user_id),
    vehicle_type      text,
    helmet_worn       text,
    road_type         text,
    speed             text,
    created_at        timestamptz default now(),
    status            text default 'intake',
    mode_of_transport text,
    time_to_hospital  text,
    collision_with    text
);

comment on column public."Accident_Record".mode_of_transport is
    'How patient was transported to hospital. Model input.';
comment on column public."Accident_Record".time_to_hospital is
    'Time taken to reach hospital as a range string. Model input.';
comment on column public."Accident_Record".collision_with is
    'What the vehicle collided with. Model input.';

create table if not exists public."Injury_Description" (
    injury_id      uuid primary key default gen_random_uuid(),
    accident_id    uuid references public."Accident_Record" (accident_id),
    site_of_injury text,
    type_of_injury text
);

create table if not exists public."Treatment" (
    treatment_id        uuid primary key default gen_random_uuid(),
    accident_id         uuid references public."Accident_Record" (accident_id),
    hospital_id         uuid references public."Hospital" (hospital_id),
    ward_number         text,
    reason              text,
    number_of_days_stay integer,
    decided_by          uuid references public."User" (user_id),
    logged_by           uuid references public."User" (user_id),
    created_at          timestamptz default now()
);

create table if not exists public."Transfer" (
    transfer_id          uuid primary key default gen_random_uuid(),
    accident_id          uuid references public."Accident_Record" (accident_id),
    from_hospital_id     uuid references public."Hospital" (hospital_id),
    to_hospital_id       uuid references public."Hospital" (hospital_id),
    authorized_by        uuid references public."User" (user_id),
    authorization_reason text,
    authorized_at        timestamptz,
    requested_by         uuid references public."User" (user_id),
    requested_at         timestamptz default now(),
    approved_by          uuid references public."User" (user_id),
    decided_at           timestamptz,
    status               text default 'pending_authorization'
);

-- One row per stay of an accident episode at a hospital.
create table if not exists public."Hospital_Patient" (
    id          uuid primary key default gen_random_uuid(),
    accident_id uuid references public."Accident_Record" (accident_id),
    hospital_id uuid references public."Hospital" (hospital_id),
    entered_at  timestamptz default now(),
    exited_at   timestamptz
);

-- actor_id is a User or a Government_Officer, so it has no foreign key.
create table if not exists public."AuditLog" (
    log_id      uuid primary key default gen_random_uuid(),
    actor_id    uuid not null,
    role        text not null,
    hospital_id uuid references public."Hospital" (hospital_id),
    action_type text not null,
    accident_id uuid references public."Accident_Record" (accident_id),
    "timestamp" timestamptz default now()
);
