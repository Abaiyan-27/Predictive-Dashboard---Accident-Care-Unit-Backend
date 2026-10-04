-- Structured details for audit entries.
--
-- Severity decisions are audited with the model prediction, the final decision
-- and the prediction they refer to. AuditLog had no place for those values, so
-- this adds one nullable column. Additive only: existing rows keep NULL and the
-- API writes entries without details until this migration is applied.

alter table public."AuditLog" add column if not exists details jsonb;

do $$ begin
    if not exists (select from pg_constraint where conname = 'audit_log_details_is_object') then
        alter table public."AuditLog"
            add constraint audit_log_details_is_object
            check (details is null or jsonb_typeof(details) = 'object');
    end if;
end $$;

notify pgrst, 'reload schema';
