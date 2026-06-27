do $$
begin
  if not exists (select 1 from pg_roles where rolname = 'anon') then
    create role anon nologin noinherit;
  end if;
  if not exists (select 1 from pg_roles where rolname = 'authenticated') then
    create role authenticated nologin noinherit;
  end if;
  if not exists (select 1 from pg_roles where rolname = 'service_role') then
    create role service_role nologin noinherit bypassrls;
  end if;
  if not exists (select 1 from pg_roles where rolname = 'authenticator') then
    create role authenticator login noinherit password 'postgres';
  end if;
  if not exists (select 1 from pg_roles where rolname = 'supabase_admin') then
    create role supabase_admin login createdb createrole replication bypassrls password 'postgres';
  end if;
  if not exists (select 1 from pg_roles where rolname = 'supabase_auth_admin') then
    create role supabase_auth_admin login noinherit createrole password 'postgres';
  end if;
  if not exists (select 1 from pg_roles where rolname = 'supabase_storage_admin') then
    create role supabase_storage_admin login noinherit createrole bypassrls password 'postgres';
  end if;
end $$;

grant anon, authenticated, service_role to authenticator;
grant anon, authenticated, service_role to supabase_storage_admin;
alter role supabase_storage_admin bypassrls;
alter role anon set search_path to storage, public, extensions;
alter role authenticated set search_path to storage, public, extensions;
alter role service_role set search_path to storage, public, extensions;
alter role authenticator set search_path to public, storage, extensions;
alter role supabase_storage_admin set search_path to storage, public, extensions;
do $$
begin
  execute format('grant create on database %I to supabase_storage_admin', current_database());
  execute format('alter role supabase_storage_admin in database %I set search_path to storage, public, extensions', current_database());
  execute format('alter role service_role in database %I set search_path to storage, public, extensions', current_database());
end $$;

create table if not exists public.threads (
  id text primary key,
  name text not null,
  embed_model text not null,
  settings jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz
);

create table if not exists public.files (
  file_hash text primary key,
  file_name text not null,
  file_path text,
  source_type text not null default 'pdf',
  file_status jsonb not null default '{}'::jsonb,
  parsed_sentences_json text,
  storage_bucket text,
  storage_path text,
  created_at timestamptz not null default now()
);

create table if not exists public.thread_files (
  thread_id text not null references public.threads(id) on delete cascade,
  file_hash text not null references public.files(file_hash) on delete cascade,
  added_at timestamptz not null default now(),
  primary key (thread_id, file_hash)
);

create table if not exists public.messages (
  id text primary key,
  thread_id text not null references public.threads(id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  context_compact text,
  reasoning text,
  reasoning_available boolean not null default false,
  reasoning_format text not null default 'none',
  web_sources jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.thread_file_annotations (
  thread_id text not null references public.threads(id) on delete cascade,
  file_hash text not null references public.files(file_hash) on delete cascade,
  annotations_json text not null default '[]',
  created_at timestamptz not null default now(),
  updated_at timestamptz,
  primary key (thread_id, file_hash)
);

create table if not exists public.thread_stats (
  thread_id text primary key references public.threads(id) on delete cascade,
  total_qa_pairs integer not null default 0,
  total_qa_chars integer not null default 0,
  avg_qa_chars double precision not null default 0,
  last_qa_at timestamptz,
  documents_meta jsonb not null default '{}'::jsonb,
  last_updated_at timestamptz not null default now()
);

create index if not exists idx_thread_created_at on public.threads(created_at);
create index if not exists idx_thread_name on public.threads(name);
create index if not exists idx_thread_embed_model on public.threads(embed_model);
create index if not exists idx_file_name on public.files(file_name);
create index if not exists idx_file_source_type on public.files(source_type);
create index if not exists idx_thread_file_file_hash on public.thread_files(file_hash);
create index if not exists idx_message_thread_created on public.messages(thread_id, created_at);
create index if not exists idx_message_role on public.messages(role);

alter table public.threads enable row level security;
alter table public.files enable row level security;
alter table public.thread_files enable row level security;
alter table public.messages enable row level security;
alter table public.thread_file_annotations enable row level security;
alter table public.thread_stats enable row level security;

do $$
begin
  create policy "anon read threads" on public.threads for select to anon using (true);
exception when duplicate_object then null;
end $$;
do $$
begin
  create policy "anon write threads" on public.threads for all to anon using (true) with check (true);
exception when duplicate_object then null;
end $$;
do $$
begin
  create policy "anon read files" on public.files for select to anon using (true);
exception when duplicate_object then null;
end $$;
do $$
begin
  create policy "anon write files" on public.files for all to anon using (true) with check (true);
exception when duplicate_object then null;
end $$;
do $$
begin
  create policy "anon read thread files" on public.thread_files for select to anon using (true);
exception when duplicate_object then null;
end $$;
do $$
begin
  create policy "anon write thread files" on public.thread_files for all to anon using (true) with check (true);
exception when duplicate_object then null;
end $$;
do $$
begin
  create policy "anon read messages" on public.messages for select to anon using (true);
exception when duplicate_object then null;
end $$;
do $$
begin
  create policy "anon write messages" on public.messages for all to anon using (true) with check (true);
exception when duplicate_object then null;
end $$;
do $$
begin
  create policy "anon read annotations" on public.thread_file_annotations for select to anon using (true);
exception when duplicate_object then null;
end $$;
do $$
begin
  create policy "anon write annotations" on public.thread_file_annotations for all to anon using (true) with check (true);
exception when duplicate_object then null;
end $$;
do $$
begin
  create policy "anon read stats" on public.thread_stats for select to anon using (true);
exception when duplicate_object then null;
end $$;
do $$
begin
  create policy "anon write stats" on public.thread_stats for all to anon using (true) with check (true);
exception when duplicate_object then null;
end $$;

grant usage on schema public to anon, authenticated, service_role;
grant select, insert, update, delete on all tables in schema public to anon, authenticated, service_role;
grant usage, select on all sequences in schema public to anon, authenticated, service_role;

do $$
begin
  alter publication supabase_realtime add table public.threads;
  alter publication supabase_realtime add table public.files;
  alter publication supabase_realtime add table public.thread_files;
  alter publication supabase_realtime add table public.messages;
  alter publication supabase_realtime add table public.thread_file_annotations;
  alter publication supabase_realtime add table public.thread_stats;
exception when undefined_object or duplicate_object then null;
end $$;
