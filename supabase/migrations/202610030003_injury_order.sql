-- Store which injury is primary (1) and which is secondary (2).
--
-- The models read "Site/Type of injury No 1" and "No 2", so the position of an
-- injury is data, not presentation. Until now it depended on the order rows
-- came back from Postgres, which is not guaranteed.
--
-- Existing rows: Injury_Description has no created_at and random UUID keys, so
-- no recorded value says which injury was entered first. The backfill uses the
-- physical row order (ctid) as the best available signal; the intake API
-- inserted the primary injury before the secondary one. Accidents that receive
-- a heuristic order are listed by the NOTICE below and should be checked by a
-- clinician. Accidents with more than two injuries abort the migration because
-- the application only models a primary and a secondary injury.

alter table public."Injury_Description" add column if not exists injury_order smallint;

do $$
declare
    over_two bigint;
    heuristic text;
begin
    select count(*) into over_two from (
        select accident_id from public."Injury_Description"
        group by accident_id having count(*) > 2) t;
    if over_two > 0 then
        raise exception 'Injury order migration aborted: % accident(s) have more than two injuries', over_two;
    end if;

    select string_agg(accident_id::text, ', ') into heuristic from (
        select accident_id from public."Injury_Description"
        where injury_order is null
        group by accident_id having count(*) = 2) t;
    if heuristic is not null then
        raise notice 'Injury order inferred from physical row order for accident(s): %', heuristic;
    end if;

    update public."Injury_Description" injury
    set injury_order = ordered.position
    from (
        select injury_id,
               row_number() over (partition by accident_id order by ctid) as position
        from public."Injury_Description"
        where injury_order is null
    ) ordered
    where injury.injury_id = ordered.injury_id
      and injury.injury_order is null
      -- Never collide with an order that is already set on the same accident.
      and not exists (
          select 1 from public."Injury_Description" other
          where other.accident_id = injury.accident_id
            and other.injury_order is not null
      );
end $$;

alter table public."Injury_Description" alter column injury_order set not null;

do $$
begin
    if not exists (select 1 from pg_constraint where conname = 'injury_order_primary_or_secondary') then
        alter table public."Injury_Description"
            add constraint injury_order_primary_or_secondary check (injury_order in (1, 2));
    end if;
    if not exists (select 1 from pg_constraint where conname = 'injury_one_per_order') then
        alter table public."Injury_Description"
            add constraint injury_one_per_order unique (accident_id, injury_order);
    end if;
end $$;

comment on column public."Injury_Description".injury_order is
    '1 = primary injury (model "No 1"), 2 = secondary injury (model "No 2").';

notify pgrst, 'reload schema';
