import { getSupabaseClient, useSupabaseRealtime } from "./client";

export function subscribeToThreadChanges(threadId: string, onChange: () => void): () => void {
  if (!useSupabaseRealtime()) return () => {};
  const client = getSupabaseClient();
  if (!client) return () => {};
  const channel = client
    .channel(`thread:${threadId}`)
    .on("postgres_changes", { event: "*", schema: "public", table: "messages", filter: `thread_id=eq.${threadId}` }, onChange)
    .on("postgres_changes", { event: "*", schema: "public", table: "thread_files", filter: `thread_id=eq.${threadId}` }, onChange)
    .on("postgres_changes", { event: "*", schema: "public", table: "thread_file_annotations", filter: `thread_id=eq.${threadId}` }, onChange)
    .subscribe();
  return () => {
    client.removeChannel(channel);
  };
}
