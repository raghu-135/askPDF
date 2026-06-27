create or replace function public.claim_file_status(
  p_file_hash text,
  p_section text,
  p_status text,
  p_started_at text default null
) returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  current_doc jsonb;
  section_doc jsonb;
  current_status text;
begin
  select coalesce(file_status, '{}'::jsonb) into current_doc
  from public.files
  where file_hash = p_file_hash
  for update;

  if not found then
    return false;
  end if;

  section_doc := coalesce(current_doc -> p_section, '{}'::jsonb);
  current_status := coalesce(section_doc ->> 'status', 'unknown');
  if current_status in ('running', 'completed') then
    return false;
  end if;

  section_doc := section_doc || jsonb_build_object('status', p_status);
  if p_started_at is not null then
    section_doc := section_doc || jsonb_build_object('started_at', p_started_at);
  end if;
  section_doc := section_doc - 'error';

  update public.files
  set file_status = current_doc
    || jsonb_build_object(p_section, section_doc)
    || jsonb_build_object(p_section || '_status', section_doc)
    || jsonb_build_object('updated_at', to_jsonb(to_char(now() at time zone 'utc', 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"')))
  where file_hash = p_file_hash;

  return true;
end;
$$;

create or replace function public.complete_parsing_atomically(
  p_file_hash text,
  p_parsed_sentences_json text,
  p_finished_at text
) returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  current_doc jsonb;
  parsing_doc jsonb;
begin
  select coalesce(file_status, '{}'::jsonb) into current_doc
  from public.files
  where file_hash = p_file_hash
  for update;

  if not found then
    return false;
  end if;

  parsing_doc := coalesce(current_doc -> 'parsing_status', current_doc -> 'parsing', '{}'::jsonb)
    || jsonb_build_object('status', 'completed', 'finished_at', p_finished_at);
  parsing_doc := parsing_doc - 'error';

  update public.files
  set parsed_sentences_json = p_parsed_sentences_json,
      file_status = current_doc
        || jsonb_build_object('parsing', parsing_doc)
        || jsonb_build_object('parsing_status', parsing_doc)
        || jsonb_build_object('updated_at', to_jsonb(to_char(now() at time zone 'utc', 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"')))
  where file_hash = p_file_hash;

  return true;
end;
$$;

create or replace function public.fail_parsing_atomically(
  p_file_hash text,
  p_error text,
  p_finished_at text
) returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  current_doc jsonb;
  parsing_doc jsonb;
begin
  select coalesce(file_status, '{}'::jsonb) into current_doc
  from public.files
  where file_hash = p_file_hash
  for update;

  if not found then
    return false;
  end if;

  parsing_doc := coalesce(current_doc -> 'parsing_status', current_doc -> 'parsing', '{}'::jsonb)
    || jsonb_build_object('status', 'failed', 'finished_at', p_finished_at, 'error', p_error);

  update public.files
  set file_status = current_doc
    || jsonb_build_object('parsing', parsing_doc)
    || jsonb_build_object('parsing_status', parsing_doc)
    || jsonb_build_object('updated_at', to_jsonb(to_char(now() at time zone 'utc', 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"')))
  where file_hash = p_file_hash;

  return true;
end;
$$;

create or replace function public.delete_message_pair(p_message_id text)
returns text[]
language plpgsql
security definer
set search_path = public
as $$
declare
  target public.messages%rowtype;
  pair_id text;
  deleted text[];
begin
  select * into target from public.messages where id = p_message_id;
  if not found then
    return array[]::text[];
  end if;

  if target.role = 'assistant' then
    select id into pair_id
    from public.messages
    where thread_id = target.thread_id
      and role = 'user'
      and created_at <= target.created_at
      and id <> p_message_id
    order by created_at desc
    limit 1;
  elsif target.role = 'user' then
    select id into pair_id
    from public.messages
    where thread_id = target.thread_id
      and role = 'assistant'
      and created_at >= target.created_at
      and id <> p_message_id
    order by created_at asc
    limit 1;
  end if;

  deleted := array_remove(array[p_message_id, pair_id], null);
  delete from public.messages where id = any(deleted);
  return deleted;
end;
$$;

create or replace function public.recompute_qa_stats(p_thread_id text)
returns void
language sql
security definer
set search_path = public
as $$
  insert into public.thread_stats (
    thread_id,
    total_qa_pairs,
    total_qa_chars,
    avg_qa_chars,
    last_qa_at,
    last_updated_at
  )
  select
    p_thread_id,
    count(*)::integer,
    coalesce(sum(length(content)), 0)::integer,
    case when count(*) = 0 then 0 else coalesce(sum(length(content)), 0)::double precision / count(*) end,
    max(created_at),
    now()
  from public.messages
  where thread_id = p_thread_id and role = 'assistant'
  on conflict (thread_id) do update
  set total_qa_pairs = excluded.total_qa_pairs,
      total_qa_chars = excluded.total_qa_chars,
      avg_qa_chars = excluded.avg_qa_chars,
      last_qa_at = excluded.last_qa_at,
      last_updated_at = now();
$$;

create or replace function public.remove_thread_file_pair(p_thread_id text, p_file_hash text)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
begin
  delete from public.thread_file_annotations
  where thread_id = p_thread_id and file_hash = p_file_hash;

  delete from public.thread_files
  where thread_id = p_thread_id and file_hash = p_file_hash;

  return found;
end;
$$;
