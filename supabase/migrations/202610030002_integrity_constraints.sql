-- Tighten integrity on the existing tables to match how the API behaves.
--
-- Safety: nothing here deletes or rewrites data. A preflight block counts every
-- row that would violate a new constraint and aborts the migration (before any
-- change) with the counts, so the data can be fixed deliberately first.
-- The one exception is noted below: staff users without a hospital are allowed
-- to remain (NOT VALID check) because the live project has such an admin.

create or replace function pg_temp.ensure_constraint(tbl regclass, cname text, ddl text)
returns void language plpgsql as $$
begin
    if not exists (select 1 from pg_constraint where conrelid = tbl and conname = cname) then
        execute format('alter table %s add constraint %I %s', tbl, cname, ddl);
    end if;
end $$;

-- ── Preflight ──────────────────────────────────────────────────────────────
do $$
declare
    problems text := '';
    n bigint;
begin
    select count(*) into n from public."Accident_Record"
        where patient_id is null or hospital_id is null or first_hospital_id is null
           or created_by is null or status is null;
    if n > 0 then problems := problems || format(E'\n  Accident_Record missing patient/hospital/first_hospital/creator/status: %s', n); end if;

    select count(*) into n from public."Injury_Description" where accident_id is null;
    if n > 0 then problems := problems || format(E'\n  Injury_Description without accident: %s', n); end if;

    select count(*) into n from public."Treatment"
        where accident_id is null or hospital_id is null or logged_by is null;
    if n > 0 then problems := problems || format(E'\n  Treatment missing accident/hospital/logged_by: %s', n); end if;

    select count(*) into n from public."Treatment" where number_of_days_stay < 0;
    if n > 0 then problems := problems || format(E'\n  Treatment with negative days: %s', n); end if;

    select count(*) into n from public."Transfer"
        where accident_id is null or from_hospital_id is null or to_hospital_id is null
           or requested_by is null or status is null or requested_at is null;
    if n > 0 then problems := problems || format(E'\n  Transfer missing accident/hospitals/requester/status: %s', n); end if;

    select count(*) into n from public."Transfer" where from_hospital_id = to_hospital_id;
    if n > 0 then problems := problems || format(E'\n  Transfer to the same hospital: %s', n); end if;

    select count(*) into n from public."Transfer"
        where status not in ('pending_authorization', 'authorized', 'approved', 'rejected');
    if n > 0 then problems := problems || format(E'\n  Transfer with unknown status: %s', n); end if;

    select count(*) into n from public."Hospital_Patient"
        where accident_id is null or hospital_id is null or entered_at is null;
    if n > 0 then problems := problems || format(E'\n  Hospital_Patient missing accident/hospital/entered_at: %s', n); end if;

    select count(*) into n from public."Hospital_Patient" where exited_at < entered_at;
    if n > 0 then problems := problems || format(E'\n  Hospital_Patient exiting before entering: %s', n); end if;

    select count(*) into n from (
        select accident_id from public."Hospital_Patient" where exited_at is null
        group by accident_id having count(*) > 1) open_stays;
    if n > 0 then problems := problems || format(E'\n  Accidents with more than one open hospital stay: %s', n); end if;

    select count(*) into n from public."User" where role not in ('nurse', 'doctor', 'admin');
    if n > 0 then problems := problems || format(E'\n  User with role outside nurse/doctor/admin: %s', n); end if;

    select count(*) into n from (
        select lower(email) from public."User" group by lower(email) having count(*) > 1) d;
    if n > 0 then problems := problems || format(E'\n  Duplicate User emails (case-insensitive): %s', n); end if;

    select count(*) into n from (
        select lower(email) from public."Government_Officer" group by lower(email) having count(*) > 1) d;
    if n > 0 then problems := problems || format(E'\n  Duplicate Government_Officer emails: %s', n); end if;

    select count(*) into n from public."AuditLog" where "timestamp" is null;
    if n > 0 then problems := problems || format(E'\n  AuditLog without timestamp: %s', n); end if;

    if problems <> '' then
        raise exception 'Integrity migration aborted; fix these rows first:%', problems;
    end if;
end $$;

-- ── Accident episodes keep their patient, hospital and creator ─────────────
alter table public."Accident_Record"
    alter column patient_id set not null,
    alter column hospital_id set not null,
    alter column first_hospital_id set not null,
    alter column created_by set not null,
    alter column status set not null;

create index if not exists accident_record_hospital_created_idx
    on public."Accident_Record" (hospital_id, created_at desc);
create index if not exists accident_record_creator_created_idx
    on public."Accident_Record" (created_by, created_at desc);
create index if not exists accident_record_patient_idx
    on public."Accident_Record" (patient_id);

-- ── Injuries belong to an accident ─────────────────────────────────────────
alter table public."Injury_Description" alter column accident_id set not null;

-- ── Treatments ────────────────────────────────────────────────────────────
alter table public."Treatment"
    alter column accident_id set not null,
    alter column hospital_id set not null,
    alter column logged_by set not null;
select pg_temp.ensure_constraint('public."Treatment"', 'treatment_days_non_negative',
    'check (number_of_days_stay is null or number_of_days_stay >= 0)');
create index if not exists treatment_accident_idx on public."Treatment" (accident_id, created_at);

-- ── Transfers ─────────────────────────────────────────────────────────────
alter table public."Transfer"
    alter column accident_id set not null,
    alter column from_hospital_id set not null,
    alter column to_hospital_id set not null,
    alter column requested_by set not null,
    alter column status set not null,
    alter column requested_at set not null;
select pg_temp.ensure_constraint('public."Transfer"', 'transfer_status_valid',
    $c$check (status in ('pending_authorization', 'authorized', 'approved', 'rejected'))$c$);
select pg_temp.ensure_constraint('public."Transfer"', 'transfer_between_different_hospitals',
    'check (from_hospital_id <> to_hospital_id)');
create index if not exists transfer_accident_idx on public."Transfer" (accident_id, requested_at desc);
create index if not exists transfer_status_idx on public."Transfer" (status);

-- ── Hospital stays ────────────────────────────────────────────────────────
alter table public."Hospital_Patient"
    alter column accident_id set not null,
    alter column hospital_id set not null,
    alter column entered_at set not null;
select pg_temp.ensure_constraint('public."Hospital_Patient"', 'hospital_patient_exit_after_entry',
    'check (exited_at is null or exited_at >= entered_at)');
create index if not exists hospital_patient_accident_idx on public."Hospital_Patient" (accident_id);
create unique index if not exists hospital_patient_one_open_stay
    on public."Hospital_Patient" (accident_id) where exited_at is null;

-- ── Staff users belong to a hospital; government officers do not ───────────
select pg_temp.ensure_constraint('public."User"', 'user_role_valid',
    $c$check (role in ('nurse', 'doctor', 'admin'))$c$);
-- NOT VALID: enforced for every new or updated row, but existing rows are not
-- re-checked. The live project has one admin with no hospital; assign one and
-- then run: alter table public."User" validate constraint user_staff_has_hospital;
select pg_temp.ensure_constraint('public."User"', 'user_staff_has_hospital',
    'check (hospital_id is not null) not valid');
create unique index if not exists user_email_unique on public."User" (lower(email));
create index if not exists user_hospital_idx on public."User" (hospital_id);
create unique index if not exists government_officer_email_unique
    on public."Government_Officer" (lower(email));

-- ── Audit log ─────────────────────────────────────────────────────────────
alter table public."AuditLog" alter column "timestamp" set not null;
create index if not exists audit_log_timestamp_idx on public."AuditLog" ("timestamp" desc);

notify pgrst, 'reload schema';
