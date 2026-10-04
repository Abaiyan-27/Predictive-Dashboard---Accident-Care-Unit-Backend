-- Decisions stay inside the hospital that holds the patient.
--
-- Replaces validate_prediction_decision() from 202610030004 with the same role
-- and source rules plus two hospital rules:
--   * the deciding user belongs to the hospital where the prediction was made;
--   * that hospital still holds the accident episode (not transferred away).
-- The API enforces the same rules (app/core/authorization.py); this makes them
-- hold for any writer, including direct SQL. No existing rows are touched:
-- the trigger only runs on new decisions.

create or replace function public.validate_prediction_decision()
returns trigger language plpgsql as $$
declare
    model_prediction    text;
    model_status        text;
    prediction_hospital uuid;
    episode_hospital    uuid;
    decider_role        text;
    decider_hospital    uuid;
    allowed_roles       text[];
begin
    select p.prediction, p.status, p.hospital_id, a.hospital_id
      into model_prediction, model_status, prediction_hospital, episode_hospital
    from public."Model_Prediction" p
    join public."Accident_Record" a on a.accident_id = p.accident_id
    where p.prediction_id = new.prediction_id;

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

notify pgrst, 'reload schema';
