-- Persistent model predictions and, separately, the human decisions made on them.
--
--   ML_Model             one row per patient model; who may decide on it
--   Model_Output_Label   each model's valid output set (and which outputs a
--                        human may choose as a final decision)
--   Model_Version        the exact artifact + configuration that produced a
--                        prediction, identified by a SHA-256 fingerprint
--   Model_Prediction     append-only: what the model said, when, with which
--                        inputs and version. Never updated or deleted.
--   Prediction_Decision  append-only: the final human decision on a prediction.
--                        A later decision is a new row; the prediction row is
--                        never touched.

-- ── Models ────────────────────────────────────────────────────────────────
create table if not exists public."ML_Model" (
    model_key      text primary key,
    display_name   text not null,
    decision_roles text[] not null check (cardinality(decision_roles) > 0)
);

-- Severity triage is a nurse decision; the three prognosis models are doctors'.
insert into public."ML_Model" (model_key, display_name, decision_roles) values
    ('severity',      'Severity triage',          array['nurse']),
    ('stay_duration', 'Length of stay',           array['doctor']),
    ('transfer',      'Transfer probability',     array['doctor']),
    ('discharge',     'Discharge outcome',        array['doctor'])
on conflict (model_key) do update
    set display_name = excluded.display_name,
        decision_roles = excluded.decision_roles;

-- ── Valid outputs per model ───────────────────────────────────────────────
-- Labels match exactly what the API returns (app/contracts/severity_labels.json
-- for severity; each model contract's label map for the others).
create table if not exists public."Model_Output_Label" (
    model_key        text not null references public."ML_Model" (model_key),
    label            text not null,
    sort_order       smallint not null,
    human_selectable boolean not null,
    primary key (model_key, label),
    unique (model_key, label, human_selectable)
);

insert into public."Model_Output_Label" (model_key, label, sort_order, human_selectable) values
    ('severity', 'HIGH', 1, true),
    ('severity', 'BORDERLINE', 2, true),
    ('severity', 'LOW', 3, true),
    -- The model's "cannot score" answer; a clinician's final decision resolves it.
    ('severity', 'DEFER TO CLINICAL JUDGEMENT', 4, false),
    ('stay_duration', '0-1 days', 1, true),
    ('stay_duration', '2-3 days', 2, true),
    ('stay_duration', '4+ days', 3, true),
    ('transfer', 'Transfer', 1, true),
    ('transfer', 'No transfer', 2, true),
    ('discharge', 'Complete recovery', 1, true),
    ('discharge', 'Partial recovery', 2, true),
    ('discharge', 'Further interventions', 3, true)
on conflict (model_key, label) do update
    set sort_order = excluded.sort_order,
        human_selectable = excluded.human_selectable;

-- ── Model versions ────────────────────────────────────────────────────────
create table if not exists public."Model_Version" (
    model_version_id uuid primary key default gen_random_uuid(),
    model_key        text not null references public."ML_Model" (model_key),
    -- SHA-256 over the model contract, the artifact and metadata files, and the
    -- backend's input gate for the model. Identical artifacts give the same id.
    fingerprint      text not null check (fingerprint ~ '^[0-9a-f]{64}$'),
    contract_name    text,
    contract_version text,          -- the "version" string in model_contract.json
    artifact_name    text not null,
    source           text not null, -- local bundle or Hugging Face repository
    first_seen_at    timestamptz not null default now(),
    unique (model_key, fingerprint),
    unique (model_version_id, model_key)
);

-- ── Predictions ───────────────────────────────────────────────────────────
create table if not exists public."Model_Prediction" (
    prediction_id    uuid primary key default gen_random_uuid(),
    -- Strict insertion order, so repeated predictions are always distinguishable.
    prediction_seq   bigint generated always as identity unique,
    accident_id      uuid not null references public."Accident_Record" (accident_id),
    -- Hospital responsible for the episode when the prediction was made.
    hospital_id      uuid not null references public."Hospital" (hospital_id),
    model_key        text not null references public."ML_Model" (model_key),
    model_version_id uuid not null,
    status           text not null
        check (status in ('OK', 'LOW_CONFIDENCE', 'REFUSED', 'UNAVAILABLE', 'ERROR')),
    prediction       text,
    probability      numeric(5, 4) check (probability between 0 and 1),
    probabilities    jsonb not null default '{}'::jsonb
        check (jsonb_typeof(probabilities) = 'object'),
    confidence_level text check (confidence_level in ('high', 'low')),
    completeness     numeric(4, 3) not null check (completeness between 0 and 1),
    message          text,
    -- The model's raw inputs exactly as passed to it (only that model's fields).
    inputs           jsonb not null check (jsonb_typeof(inputs) = 'object'),
    inputs_hash      text not null check (inputs_hash ~ '^[0-9a-f]{64}$'),
    trigger_event    text not null check (trigger_event in (
        'intake_created', 'intake_updated', 'treatment_recorded', 'prognosis_viewed')),
    requested_by     uuid references public."User" (user_id),
    predicted_at     timestamptz not null default now(),

    constraint model_prediction_id_model_key unique (prediction_id, model_key),
    constraint model_prediction_version_matches_model
        foreign key (model_version_id, model_key)
        references public."Model_Version" (model_version_id, model_key),
    constraint model_prediction_label_valid
        foreign key (model_key, prediction)
        references public."Model_Output_Label" (model_key, label),
    -- Severity always answers with a label (DEFER when it cannot score); the
    -- other models give no label unless they scored.
    constraint model_prediction_outcome_consistent check (
        case when model_key = 'severity' then
            prediction is not null
            and (status in ('OK', 'LOW_CONFIDENCE'))
                = (prediction <> 'DEFER TO CLINICAL JUDGEMENT')
        else
            (status in ('OK', 'LOW_CONFIDENCE')) = (prediction is not null)
        end
    )
);

create index if not exists model_prediction_history_idx
    on public."Model_Prediction" (accident_id, model_key, prediction_seq desc);
create index if not exists model_prediction_version_idx
    on public."Model_Prediction" (model_version_id);

-- ── Human decisions ───────────────────────────────────────────────────────
create table if not exists public."Prediction_Decision" (
    decision_id      uuid primary key default gen_random_uuid(),
    decision_seq     bigint generated always as identity unique,
    prediction_id    uuid not null,
    model_key        text not null,
    final_decision   text not null,
    -- Lets the label foreign key require a human-selectable output.
    human_selectable boolean not null generated always as (true) stored,
    -- MODEL  = the clinician accepted the model's prediction as the decision
    -- NURSE  = a nurse overrode it; DOCTOR = a doctor overrode it
    decision_source  text not null check (decision_source in ('MODEL', 'NURSE', 'DOCTOR')),
    decided_by       uuid not null references public."User" (user_id),
    decided_at       timestamptz not null default now(),
    reason           text,

    constraint prediction_decision_prediction
        foreign key (prediction_id, model_key)
        references public."Model_Prediction" (prediction_id, model_key),
    constraint prediction_decision_label_valid
        foreign key (model_key, final_decision, human_selectable)
        references public."Model_Output_Label" (model_key, label, human_selectable)
);

create index if not exists prediction_decision_history_idx
    on public."Prediction_Decision" (prediction_id, decision_seq desc);
create index if not exists prediction_decision_decided_by_idx
    on public."Prediction_Decision" (decided_by);

-- Rules that span tables: who may decide, and what each source means.
create or replace function public.validate_prediction_decision()
returns trigger language plpgsql as $$
declare
    model_prediction text;
    model_status     text;
    decider_role     text;
    allowed_roles    text[];
begin
    select prediction, status into model_prediction, model_status
    from public."Model_Prediction" where prediction_id = new.prediction_id;

    select role into decider_role from public."User" where user_id = new.decided_by;
    select decision_roles into allowed_roles from public."ML_Model" where model_key = new.model_key;

    if decider_role is null or not (decider_role = any (allowed_roles)) then
        raise exception 'A % cannot record % decisions (allowed: %)',
            coalesce(decider_role, 'unknown user'), new.model_key, allowed_roles
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

drop trigger if exists prediction_decision_validate on public."Prediction_Decision";
create trigger prediction_decision_validate
    before insert on public."Prediction_Decision"
    for each row execute function public.validate_prediction_decision();

-- ── Append-only history ───────────────────────────────────────────────────
create or replace function public.forbid_history_change()
returns trigger language plpgsql as $$
begin
    raise exception '% is append-only history; % is not allowed', tg_table_name, tg_op
        using errcode = '42501';
end $$;

drop trigger if exists model_prediction_append_only on public."Model_Prediction";
create trigger model_prediction_append_only
    before update or delete on public."Model_Prediction"
    for each row execute function public.forbid_history_change();

drop trigger if exists prediction_decision_append_only on public."Prediction_Decision";
create trigger prediction_decision_append_only
    before update or delete on public."Prediction_Decision"
    for each row execute function public.forbid_history_change();

-- ── Current view ──────────────────────────────────────────────────────────
-- Latest prediction per accident and model, with its latest decision (if any).
create or replace view public."Current_Model_Prediction"
with (security_invoker = true) as
select distinct on (p.accident_id, p.model_key)
    p.prediction_id,
    p.prediction_seq,
    p.accident_id,
    p.hospital_id,
    p.model_key,
    p.model_version_id,
    p.status,
    p.prediction,
    p.probability,
    p.completeness,
    p.predicted_at,
    d.decision_id,
    d.final_decision,
    d.decision_source,
    d.decided_by,
    d.decided_at
from public."Model_Prediction" p
left join lateral (
    select * from public."Prediction_Decision" decision
    where decision.prediction_id = p.prediction_id
    order by decision.decision_seq desc
    limit 1
) d on true
order by p.accident_id, p.model_key, p.prediction_seq desc;

-- The API uses the service role, which bypasses RLS. Enabling RLS with no
-- policies keeps these clinical tables closed to the anon/authenticated keys.
alter table public."ML_Model" enable row level security;
alter table public."Model_Output_Label" enable row level security;
alter table public."Model_Version" enable row level security;
alter table public."Model_Prediction" enable row level security;
alter table public."Prediction_Decision" enable row level security;

notify pgrst, 'reload schema';
