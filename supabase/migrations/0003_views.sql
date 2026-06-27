create or replace view public.thread_list_view as
select
  t.id,
  t.name,
  t.embed_model,
  t.settings,
  t.created_at,
  t.updated_at,
  count(distinct m.id)::integer as message_count,
  count(distinct tf.file_hash)::integer as file_count,
  max(m.created_at) as last_message_at,
  greatest(
    coalesce(max(m.created_at), '-infinity'::timestamptz),
    coalesce(t.updated_at, t.created_at)
  ) as latest_activity_at
from public.threads t
left join public.messages m on m.thread_id = t.id
left join public.thread_files tf on tf.thread_id = t.id
group by t.id;

create or replace view public.thread_files_view as
select
  tf.thread_id,
  f.file_hash,
  f.file_name,
  f.file_path,
  f.source_type,
  f.file_status,
  f.parsed_sentences_json,
  f.storage_bucket,
  f.storage_path,
  f.created_at,
  tf.added_at
from public.thread_files tf
join public.files f on f.file_hash = tf.file_hash;

create or replace view public.thread_detail_view as
select
  t.id,
  t.name,
  t.embed_model,
  t.settings,
  t.created_at,
  t.updated_at,
  coalesce(ts.total_qa_pairs, 0) as total_qa_pairs,
  coalesce(ts.total_qa_chars, 0) as total_qa_chars,
  coalesce(ts.avg_qa_chars, 0) as avg_qa_chars,
  ts.last_qa_at,
  coalesce(ts.documents_meta, '{}'::jsonb) as documents_meta,
  count(distinct tf.file_hash)::integer as file_count,
  count(distinct m.id)::integer as message_count
from public.threads t
left join public.thread_stats ts on ts.thread_id = t.id
left join public.thread_files tf on tf.thread_id = t.id
left join public.messages m on m.thread_id = t.id
group by t.id, ts.thread_id;

grant select on public.thread_list_view to anon, authenticated, service_role;
grant select on public.thread_files_view to anon, authenticated, service_role;
grant select on public.thread_detail_view to anon, authenticated, service_role;
