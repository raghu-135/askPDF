import { getSupabaseClient } from "./client";
import type { SupabaseThreadRow } from "./types";

export async function listSupabaseThreads(): Promise<{ threads: SupabaseThreadRow[] }> {
  const client = getSupabaseClient();
  if (!client) throw new Error("Supabase client is not configured");
  const { data, error } = await client
    .from("thread_list_view")
    .select("id,name,embed_model,settings,created_at,updated_at,message_count,file_count,last_message_at,latest_activity_at")
    .order("latest_activity_at", { ascending: false });
  if (error) throw error;
  return { threads: (data || []) as SupabaseThreadRow[] };
}

export async function getSupabaseThread(threadId: string): Promise<SupabaseThreadRow | null> {
  const client = getSupabaseClient();
  if (!client) throw new Error("Supabase client is not configured");
  const { data, error } = await client
    .from("thread_detail_view")
    .select("*")
    .eq("id", threadId)
    .maybeSingle();
  if (error) throw error;
  return data as SupabaseThreadRow | null;
}
