do $$
begin
  if exists (select 1 from information_schema.tables where table_schema = 'storage' and table_name = 'buckets') then
    grant usage on schema storage to anon, authenticated, service_role, supabase_storage_admin;
    grant select on storage.buckets to anon, authenticated, service_role;
    grant all privileges on storage.buckets to supabase_storage_admin;
    insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
    values ('pdfs', 'pdfs', false, 52428800, array['application/pdf'])
    on conflict (id) do update
    set name = excluded.name,
        public = excluded.public,
        file_size_limit = excluded.file_size_limit,
        allowed_mime_types = excluded.allowed_mime_types;
  end if;
end $$;

do $$
begin
  if exists (select 1 from information_schema.tables where table_schema = 'storage' and table_name = 'objects') then
    alter table storage.objects enable row level security;
    grant usage on schema storage to anon, authenticated, service_role, supabase_storage_admin;
    grant select on storage.objects to anon, authenticated;
    grant all privileges on storage.objects to service_role, supabase_storage_admin;
  end if;
end $$;

do $$
begin
  if exists (select 1 from information_schema.tables where table_schema = 'storage' and table_name = 'objects') then
    create policy "anon read pdf objects" on storage.objects
      for select to anon
      using (bucket_id = 'pdfs');
  end if;
exception when duplicate_object then null;
end $$;

do $$
begin
  if exists (select 1 from information_schema.tables where table_schema = 'storage' and table_name = 'objects') then
    create policy "service role manage pdf objects" on storage.objects
      for all to service_role
      using (bucket_id = 'pdfs')
      with check (bucket_id = 'pdfs');
  end if;
exception when duplicate_object then null;
end $$;
