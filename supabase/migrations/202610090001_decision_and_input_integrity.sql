-- Decision finality and input locks, enforced by the database.
--
-- The API already applies these rules, but as "check, then write" in separate
-- requests, so two concurrent requests could both pass the check. This makes
-- the database the final authority, safe across any number of API workers:
--
--   * A prediction has at most one decision (unique index), and an episode has
--     at most one decision per model, made on its latest prediction for that
--     model. Concurrent decisions on one episode are serialised by locking the
--     episode row; the loser gets SQLSTATE 23505 (the API answers 409).
--   * Severity inputs cannot change once the Severity decision exists, and the
--     inputs read only by the doctor models cannot change once treatment is
--     recorded (the same fields the API locks). Re-saving an identical value is
--     allowed. Violations raise SQLSTATE 23514 (the API answers 409).
--   * Treatment cannot be added once any doctor-model decision exists.
--
-- Nothing here changes or deletes existing rows. The unique index is created
-- only when existing data already satisfies it; otherwise a notice is raised
-- and the trigger below still blocks any further duplicate.

-- ── One decision per prediction ───────────────────────────────────────────
do $$
begin
    if exists (
        select 1 from public."Prediction_Decision"
        group by prediction_id having count(*) > 1
    ) then
        raise notice 'Prediction_Decision has predictions with more than one decision; '
                     'prediction_decision_one_per_prediction was not created. '
                     'The decision trigger still refuses new duplicates.';
    else
        create unique index if not exists prediction_decision_one_per_prediction
            on public."Prediction_Decision" (prediction_id);
    end if;
end $$;

-- ── Decision rules (replaces the 202610040001 version) ─────────────────────
create or replace function public.validate_prediction_decision()
returns trigger language plpgsql as $$
declare
    model_prediction    text;
    model_status        text;
    prediction_hospital uuid;
    prediction_accident uuid;
    prediction_seq_no   bigint;
    episode_hospital    uuid;
    decider_role        text;
    decider_hospital    uuid;
    allowed_roles       text[];
begin
    select p.prediction, p.status, p.hospital_id, p.accident_id, p.prediction_seq
      into model_prediction, model_status, prediction_hospital, prediction_accident, prediction_seq_no
    from public."Model_Prediction" p
    where p.prediction_id = new.prediction_id;

    -- Serialise every decision (and input change) on this episode.
    select a.hospital_id into episode_hospital
    from public."Accident_Record" a
    where a.accident_id = prediction_accident
    for update;

    select role, hospital_id into decider_role, decider_hospital
    from public."User" where user_id = new.decided_by;
    select decision_roles into allowed_roles from public."ML_Model" where model_key = new.model_key;

    if decider_role is null or not (decider_role = any (allowed_roles)) then
        raise exception 'A % cannot record % decisions (allowed: %)',
            coalesce(decider_role, 'unknown user'), new.model_key, allowed_roles
            using errcode = '23514';
    end if;

    if decider_hospital is null or decider_hospital is distinct from prediction_hospital then
        raise exception 'Decisions can only be recorded by staff of the hospital that made the prediction'
            using errcode = '23514';
    end if;
    if episode_hospital is distinct from prediction_hospital then
        raise exception 'The episode has moved to another hospital; decide on a prediction made there'
            using errcode = '23514';
    end if;

    -- Authorised actor: now the finality rules.
    if exists (
        select 1
        from public."Prediction_Decision" d
        join public."Model_Prediction" p on p.prediction_id = d.prediction_id
        where p.accident_id = prediction_accident and d.model_key = new.model_key
    ) then
        raise exception 'A final % decision has already been recorded for this patient', new.model_key
            using errcode = '23505';
    end if;

    if exists (
        select 1 from public."Model_Prediction" p
        where p.accident_id = prediction_accident and p.model_key = new.model_key
          and p.prediction_seq > prediction_seq_no
    ) then
        raise exception 'A newer % prediction exists for this patient; decide on that one', new.model_key
            using errcode = '23514';
    end if;

    if new.decision_source = 'MODEL' then
        if model_prediction is null or model_status not in ('OK', 'LOW_CONFIDENCE') then
            raise exception 'Prediction % has no model output to accept', new.prediction_id
                using errcode = '23514';
        end if;
        if new.final_decision <> model_prediction then
            raise exception 'decision_source MODEL requires final_decision to equal the prediction (%)',
                model_prediction using errcode = '23514';
        end if;
    else
        if new.decision_source <> upper(decider_role) then
            raise exception 'decision_source % does not match the deciding user''s role (%)',
                new.decision_source, decider_role using errcode = '23514';
        end if;
        if new.final_decision is not distinct from model_prediction then
            raise exception 'final_decision equals the model prediction; record it with decision_source MODEL'
                using errcode = '23514';
        end if;
    end if;
    return new;
end $$;

-- ── Lock helpers ──────────────────────────────────────────────────────────
create or replace function public.episode_severity_decided(p_accident uuid)
returns boolean language sql stable as $$
    select exists (
        select 1
        from public."Prediction_Decision" d
        join public."Model_Prediction" p on p.prediction_id = d.prediction_id
        where p.accident_id = p_accident and d.model_key = 'severity'
    )
$$;

create or replace function public.episode_treated(p_accident uuid)
returns boolean language sql stable as $$
    select exists (select 1 from public."Treatment" t where t.accident_id = p_accident)
$$;

create or replace function public.episode_doctor_decided(p_accident uuid)
returns boolean language sql stable as $$
    select exists (
        select 1
        from public."Prediction_Decision" d
        join public."Model_Prediction" p on p.prediction_id = d.prediction_id
        where p.accident_id = p_accident and d.model_key in ('stay_duration', 'transfer', 'discharge')
    )
$$;

-- ── Accident_Record: Severity inputs and doctor-model inputs ──────────────
create or replace function public.enforce_accident_input_locks()
returns trigger language plpgsql as $$
begin
    if (new.incident_at, new.mode_of_travel_during_accident, new.collision_with, new.vehicle_type,
        new.helmet_worn, new.passenger_type, new.mode_of_transport, new.time_to_hospital, new.other_injury)
       is distinct from
       (old.incident_at, old.mode_of_travel_during_accident, old.collision_with, old.vehicle_type,
        old.helmet_worn, old.passenger_type, old.mode_of_transport, old.time_to_hospital, old.other_injury)
       and public.episode_severity_decided(old.accident_id) then
        raise exception 'Severity inputs are locked after the Severity decision'
            using errcode = '23514';
    end if;
    if (new.road_type, new.speed, new.road_condition, new.visibility, new.road_signals_exist,
        new.collision_force_from, new.alcohol_consumption, new.illicit_drugs)
       is distinct from
       (old.road_type, old.speed, old.road_condition, old.visibility, old.road_signals_exist,
        old.collision_force_from, old.alcohol_consumption, old.illicit_drugs)
       and public.episode_treated(old.accident_id) then
        raise exception 'Inputs for the doctor''s models are locked after treatment is recorded'
            using errcode = '23514';
    end if;
    return new;
end $$;

drop trigger if exists accident_input_locks on public."Accident_Record";
create trigger accident_input_locks
    before update on public."Accident_Record"
    for each row execute function public.enforce_accident_input_locks();

-- ── Patient: age is a Severity input ──────────────────────────────────────
create or replace function public.enforce_patient_input_locks()
returns trigger language plpgsql as $$
declare
    episode uuid;
begin
    if new.age is distinct from old.age then
        for episode in
            select accident_id from public."Accident_Record"
            where patient_id = old.patient_id
            for update
        loop
            if public.episode_severity_decided(episode) then
                raise exception 'Age is a Severity input and is locked after the Severity decision'
                    using errcode = '23514';
            end if;
        end loop;
    end if;
    return new;
end $$;

drop trigger if exists patient_input_locks on public."Patient";
create trigger patient_input_locks
    before update on public."Patient"
    for each row execute function public.enforce_patient_input_locks();

-- ── Injury_Description: injury sites (Severity) and types (doctor models) ─
create or replace function public.enforce_injury_input_locks()
returns trigger language plpgsql as $$
declare
    episode    uuid := coalesce(new.accident_id, old.accident_id);
    old_site   text := case when tg_op in ('UPDATE', 'DELETE') then old.site_of_injury end;
    new_site   text := case when tg_op in ('INSERT', 'UPDATE') then new.site_of_injury end;
    old_type   text := case when tg_op in ('UPDATE', 'DELETE') then old.type_of_injury end;
    new_type   text := case when tg_op in ('INSERT', 'UPDATE') then new.type_of_injury end;
begin
    perform 1 from public."Accident_Record" where accident_id = episode for update;
    if tg_op = 'INSERT' then
        -- An upsert fires this before its conflict check: compare with the
        -- row already in that slot, if there is one.
        select i.site_of_injury, i.type_of_injury into old_site, old_type
        from public."Injury_Description" i
        where i.accident_id = new.accident_id and i.injury_order = new.injury_order;
    end if;
    if new_site is distinct from old_site and public.episode_severity_decided(episode) then
        raise exception 'Injury sites are Severity inputs and are locked after the Severity decision'
            using errcode = '23514';
    end if;
    if new_type is distinct from old_type and public.episode_treated(episode) then
        raise exception 'Injury types are locked after treatment is recorded'
            using errcode = '23514';
    end if;
    return case when tg_op = 'DELETE' then old else new end;
end $$;

drop trigger if exists injury_input_locks on public."Injury_Description";
create trigger injury_input_locks
    before insert or update or delete on public."Injury_Description"
    for each row execute function public.enforce_injury_input_locks();

-- ── Treatment: none after a doctor-model decision ─────────────────────────
create or replace function public.enforce_treatment_lock()
returns trigger language plpgsql as $$
begin
    perform 1 from public."Accident_Record" where accident_id = new.accident_id for update;
    if public.episode_doctor_decided(new.accident_id) then
        raise exception 'Treatment is locked: a doctor-model decision has already been recorded'
            using errcode = '23514';
    end if;
    return new;
end $$;

drop trigger if exists treatment_lock on public."Treatment";
create trigger treatment_lock
    before insert on public."Treatment"
    for each row execute function public.enforce_treatment_lock();

-- The lock helpers are read-only and run with the caller's rights, so under
-- RLS the public roles see no rows through them. They are left executable so
-- the triggers work for whichever role the backend connects as.

notify pgrst, 'reload schema';
