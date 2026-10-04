-- The hospital network. Seeds the eight network hospitals and marks them so the
-- next phase can enforce network membership. Hospitals already in the table
-- (including any outside the network) are kept unchanged.

do $$
declare dupes text;
begin
    select string_agg(name, ', ') into dupes from (
        select name from public."Hospital" group by name having count(*) > 1) d;
    if dupes is not null then
        raise exception 'Hospital network migration aborted: duplicate hospital names: %', dupes;
    end if;
end $$;

create unique index if not exists hospital_name_unique on public."Hospital" (name);

alter table public."Hospital"
    add column if not exists is_network_hospital boolean not null default false;

insert into public."Hospital" (name, type, region, district, is_network_hospital) values
    ('DGH – Vavuniya',                    'District General Hospital', 'Northern Province', 'Vavuniya',    true),
    ('DGH – Kilinochchi',                 'District General Hospital', 'Northern Province', 'Kilinochchi', true),
    ('Teaching Hospital – Jaffna (THJ)',  'Teaching Hospital',         'Northern Province', 'Jaffna',      true),
    ('DGH – Mullaithivu',                 'District General Hospital', 'Northern Province', 'Mullaithivu', true),
    ('Base Hospital (A) – Tellipalai',     'Base Hospital (A)',         'Northern Province', 'Jaffna',      true),
    ('Base Hospital (B) – Chavakachcheri', 'Base Hospital (B)',         'Northern Province', 'Jaffna',      true),
    ('Base Hospital (A) – Point Pedro',    'Base Hospital (A)',         'Northern Province', 'Jaffna',      true),
    ('DGH – Mannar',                      'District General Hospital', 'Northern Province', 'Mannar',      true)
on conflict (name) do update set is_network_hospital = true;

notify pgrst, 'reload schema';
