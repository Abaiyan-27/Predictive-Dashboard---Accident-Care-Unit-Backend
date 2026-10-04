-- Transfer-request workflow: who did what, when, and only valid status changes.
--
-- The four existing statuses are kept and given exact meanings:
--   pending_authorization  requested by an admin of the sending hospital
--   authorized             authorized by an admin of the sending hospital
--   approved               a nurse of the sending hospital confirmed the patient
--                          was physically transferred (the episode moves)
--   rejected               refused by a sending-hospital admin, or a nurse
--                          recorded that the transfer did not happen
-- Valid changes: pending_authorization -> authorized | rejected,
--                authorized -> approved | rejected. approved/rejected are final.
--
-- Why a schema change is needed: the Transfer table had no request message, no
-- actor for a rejection (approved_by was being reused) and no record of who
-- physically processed the transfer or when. The new columns are nullable, so
-- existing rows are unchanged. approved_by stays for rows written before this
-- migration; new rows record the processing nurse in processed_by.

alter table public."Transfer"
    add column if not exists request_message text,
    add column if not exists rejected_by uuid references public."User" (user_id),
    add column if not exists rejection_reason text,
    add column if not exists processed_by uuid references public."User" (user_id),
    add column if not exists processed_at timestamptz,
    add column if not exists processing_note text;

-- ── One open transfer per episode ─────────────────────────────────────────
-- Created only when existing data already satisfies it, so this migration can
-- never fail on (or modify) production rows. The API enforces the same rule.
do $$
declare duplicates int;
begin
    select count(*) into duplicates from (
        select accident_id from public."Transfer"
        where status in ('pending_authorization', 'authorized')
        group by accident_id having count(*) > 1) d;
    if duplicates = 0 then
        create unique index if not exists transfer_one_open_per_episode
            on public."Transfer" (accident_id) where status in ('pending_authorization', 'authorized');
    else
        raise notice 'transfer_one_open_per_episode not created: % episodes already have several open transfers', duplicates;
    end if;
end $$;

-- ── New requests go to a network hospital ─────────────────────────────────
create or replace function public.validate_transfer_insert()
returns trigger language plpgsql as $$
begin
    if not exists (select 1 from public."Hospital"
                   where hospital_id = new.to_hospital_id and is_network_hospital) then
        raise exception 'The destination must be a network hospital' using errcode = 'PT422';
    end if;
    if new.status <> 'pending_authorization' then
        raise exception 'A new transfer starts as pending_authorization' using errcode = 'PT422';
    end if;
    return new;
end $$;

drop trigger if exists transfer_validate_insert on public."Transfer";
create trigger transfer_validate_insert
    before insert on public."Transfer"
    for each row execute function public.validate_transfer_insert();

-- ── Only valid status changes, by any writer ──────────────────────────────
create or replace function public.validate_transfer_update()
returns trigger language plpgsql as $$
begin
    if new.status is distinct from old.status and not (
        (old.status = 'pending_authorization' and new.status in ('authorized', 'rejected'))
        or (old.status = 'authorized' and new.status in ('approved', 'rejected'))
    ) then
        raise exception 'A transfer cannot go from % to %', old.status, new.status using errcode = 'PT409';
    end if;
    return new;
end $$;

drop trigger if exists transfer_validate_update on public."Transfer";
create trigger transfer_validate_update
    before update on public."Transfer"
    for each row execute function public.validate_transfer_update();

-- ── The single place a transfer changes status ────────────────────────────
-- One transaction with the row locked: concurrent clicks cannot both succeed,
-- every timestamp is the database's now(), and completing a transfer moves the
-- episode and its hospital stay together. Errors use PostgREST's PTnnn codes,
-- which become that HTTP status.
create or replace function public.transfer_transition(
    p_transfer_id       uuid,
    p_action            text,     -- authorize | reject | complete | not_transferred
    p_actor             uuid,
    p_note              text default null,
    p_state_of_transfer text default null
) returns setof public."Transfer" language plpgsql as $$
declare
    t               public."Transfer"%rowtype;
    actor_role      text;
    actor_hospital  uuid;
    expected_role   text;
    episode_hospital uuid;
begin
    select * into t from public."Transfer" where transfer_id = p_transfer_id for update;
    if not found then
        raise exception 'Transfer request not found' using errcode = 'PT404';
    end if;

    expected_role := case
        when p_action in ('authorize', 'reject') then 'admin'
        when p_action in ('complete', 'not_transferred') then 'nurse'
    end;
    if expected_role is null then
        raise exception 'Unknown transfer action %', p_action using errcode = 'PT422';
    end if;
    select role, hospital_id into actor_role, actor_hospital from public."User" where user_id = p_actor;
    if actor_role is distinct from expected_role or actor_hospital is distinct from t.from_hospital_id then
        raise exception 'Only a % of the sending hospital can % this transfer', expected_role, p_action
            using errcode = 'PT403';
    end if;

    if p_action = 'authorize' and t.status = 'pending_authorization' then
        update public."Transfer"
           set status = 'authorized', authorized_by = p_actor, authorized_at = now(),
               authorization_reason = coalesce(p_note, authorization_reason)
         where transfer_id = p_transfer_id;

    elsif p_action = 'reject' and t.status in ('pending_authorization', 'authorized') then
        update public."Transfer"
           set status = 'rejected', rejected_by = p_actor, rejection_reason = p_note, decided_at = now()
         where transfer_id = p_transfer_id;

    elsif p_action = 'complete' and t.status = 'authorized' then
        select hospital_id into episode_hospital from public."Accident_Record"
         where accident_id = t.accident_id for update;
        if episode_hospital is distinct from t.from_hospital_id then
            raise exception 'The episode is no longer held by the sending hospital' using errcode = 'PT409';
        end if;
        update public."Transfer"
           set status = 'approved', processed_by = p_actor, processed_at = now(), decided_at = now(),
               processing_note = p_note, state_of_transfer = coalesce(p_state_of_transfer, state_of_transfer)
         where transfer_id = p_transfer_id;
        -- The destination now holds the episode; the old stay is closed, not removed.
        update public."Accident_Record" set hospital_id = t.to_hospital_id where accident_id = t.accident_id;
        update public."Hospital_Patient" set exited_at = now()
         where accident_id = t.accident_id and exited_at is null;
        insert into public."Hospital_Patient" (accident_id, hospital_id, entered_at)
        values (t.accident_id, t.to_hospital_id, now());

    elsif p_action = 'not_transferred' and t.status = 'authorized' then
        update public."Transfer"
           set status = 'rejected', processed_by = p_actor, processed_at = now(), decided_at = now(),
               processing_note = p_note
         where transfer_id = p_transfer_id;

    else
        raise exception 'Cannot % a transfer that is %', replace(p_action, '_', ' '), t.status
            using errcode = 'PT409';
    end if;

    return query select * from public."Transfer" where transfer_id = p_transfer_id;
end $$;

-- Only the backend (service role) may call it: never the public/anon keys.
revoke all on function public.transfer_transition(uuid, text, uuid, text, text) from public;
do $$ begin
    if exists (select from pg_roles where rolname = 'anon') then
        revoke all on function public.transfer_transition(uuid, text, uuid, text, text) from anon;
    end if;
    if exists (select from pg_roles where rolname = 'authenticated') then
        revoke all on function public.transfer_transition(uuid, text, uuid, text, text) from authenticated;
    end if;
    if exists (select from pg_roles where rolname = 'service_role') then
        grant execute on function public.transfer_transition(uuid, text, uuid, text, text) to service_role;
    end if;
end $$;

notify pgrst, 'reload schema';
