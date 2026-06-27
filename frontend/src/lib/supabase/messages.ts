import { getSupabaseClient } from "./client";
import type { SupabaseMessageRow } from "./types";

export async function getSupabaseThreadMessages(
  threadId: string,
  limit: number = 100,
  offset: number = 0
): Promise<{ messages: SupabaseMessageRow[] }> {
  const client = getSupabaseClient();
  if (!client) throw new Error("Supabase client is not configured");
  const from = offset;
  const to = offset + limit - 1;
  const { data, error } = await client
    .from("messages")
    .select("id,thread_id,role,content,context_compact,reasoning,reasoning_available,reasoning_format,web_sources,created_at")
    .eq("thread_id", threadId)
    .order("created_at", { ascending: true })
    .range(from, to);
  if (error) throw error;
  return { messages: (data || []) as SupabaseMessageRow[] };
}
