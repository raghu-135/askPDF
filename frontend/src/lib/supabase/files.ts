import { getSupabaseClient } from "./client";
import type { SupabaseAnnotationRow, SupabaseFileRow } from "./types";

export async function getSupabaseThreadFiles(threadId: string): Promise<{ files: SupabaseFileRow[] }> {
  const client = getSupabaseClient();
  if (!client) throw new Error("Supabase client is not configured");
  const { data, error } = await client
    .from("thread_files_view")
    .select("file_hash,file_name,file_path,source_type,file_status,parsed_sentences_json,storage_bucket,storage_path,created_at,added_at")
    .eq("thread_id", threadId)
    .order("added_at", { ascending: true });
  if (error) throw error;
  return { files: (data || []) as SupabaseFileRow[] };
}

export async function getSupabaseFileForThread(threadId: string, fileHash: string): Promise<SupabaseFileRow | null> {
  const client = getSupabaseClient();
  if (!client) throw new Error("Supabase client is not configured");
  const { data, error } = await client
    .from("thread_files_view")
    .select("file_hash,file_name,file_path,source_type,file_status,parsed_sentences_json,storage_bucket,storage_path,created_at,added_at")
    .eq("thread_id", threadId)
    .eq("file_hash", fileHash)
    .maybeSingle();
  if (error) throw error;
  return data as SupabaseFileRow | null;
}

export async function getSupabaseParsedSentences(fileHash: string): Promise<{ version: string; sentences: any[] }> {
  const client = getSupabaseClient();
  if (!client) throw new Error("Supabase client is not configured");
  const { data, error } = await client.from("files").select("parsed_sentences_json").eq("file_hash", fileHash).maybeSingle();
  if (error) throw error;
  const raw = data?.parsed_sentences_json;
  if (!raw) return { version: "1.0", sentences: [] };
  try {
    const parsed = JSON.parse(raw);
    return { version: parsed.version || "1.0", sentences: parsed.sentences || [] };
  } catch {
    return { version: "1.0", sentences: [] };
  }
}

export async function getSupabaseFileStatus(fileHash: string): Promise<Record<string, any> | null> {
  const client = getSupabaseClient();
  if (!client) throw new Error("Supabase client is not configured");
  const { data, error } = await client.from("files").select("file_status").eq("file_hash", fileHash).maybeSingle();
  if (error) throw error;
  return (data?.file_status || null) as Record<string, any> | null;
}

export async function getSupabaseThreadFileAnnotations(
  threadId: string,
  fileHash: string
): Promise<SupabaseAnnotationRow | null> {
  const client = getSupabaseClient();
  if (!client) throw new Error("Supabase client is not configured");
  const { data, error } = await client
    .from("thread_file_annotations")
    .select("thread_id,file_hash,annotations_json,created_at,updated_at")
    .eq("thread_id", threadId)
    .eq("file_hash", fileHash)
    .maybeSingle();
  if (error) throw error;
  return data as SupabaseAnnotationRow | null;
}
