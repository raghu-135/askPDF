create table if not exists public.migration_runs (
  id uuid primary key default (md5(random()::text || clock_timestamp()::text)::uuid),
  mode text not null,
  started_at timestamptz not null default now(),
  finished_at timestamptz,
  source_url_redacted text,
  target_url_redacted text,
  status text not null default 'running',
  summary jsonb not null default '{}'::jsonb,
  error text
);

create table if not exists public.migration_table_progress (
  run_id uuid not null references public.migration_runs(id) on delete cascade,
  table_name text not null,
  rows_read bigint not null default 0,
  rows_inserted bigint not null default 0,
  rows_updated bigint not null default 0,
  rows_skipped bigint not null default 0,
  checksum text,
  status text not null default 'pending',
  updated_at timestamptz not null default now(),
  primary key (run_id, table_name)
);

create table if not exists public.migration_validation_results (
  run_id uuid not null references public.migration_runs(id) on delete cascade,
  table_name text not null,
  source_count bigint,
  target_count bigint,
  source_checksum text,
  target_checksum text,
  sample_mismatches jsonb not null default '[]'::jsonb,
  status text not null,
  checked_at timestamptz not null default now(),
  primary key (run_id, table_name)
);

grant select, insert, update, delete on public.migration_runs to service_role;
grant select, insert, update, delete on public.migration_table_progress to service_role;
grant select, insert, update, delete on public.migration_validation_results to service_role;
