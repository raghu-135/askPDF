import { getSupabaseClient } from "./client";

export async function getSupabasePdfUrl(bucket: string | null | undefined, path: string | null | undefined): Promise<string | null> {
  if (!bucket || !path) return null;
  const client = getSupabaseClient();
  if (!client) return null;
  const { data, error } = await client.storage.from(bucket).createSignedUrl(path, 60 * 60);
  if (error) return null;
  return data.signedUrl;
}
