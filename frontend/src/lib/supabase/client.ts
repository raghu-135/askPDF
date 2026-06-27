import { createClient, type SupabaseClient } from "@supabase/supabase-js";

let cachedClient: SupabaseClient | null = null;

export function getSupabaseClient(): SupabaseClient | null {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !anonKey) return null;
  if (!cachedClient) {
    cachedClient = createClient(url, anonKey, {
      auth: { persistSession: false, autoRefreshToken: false },
      realtime: { params: { eventsPerSecond: 10 } },
    });
  }
  return cachedClient;
}

export function isSupabaseFlagEnabled(name: string): boolean {
  switch (name) {
    case "NEXT_PUBLIC_USE_SUPABASE_THREADS":
      return process.env.NEXT_PUBLIC_USE_SUPABASE_THREADS === "true";
    case "NEXT_PUBLIC_USE_SUPABASE_MESSAGES":
      return process.env.NEXT_PUBLIC_USE_SUPABASE_MESSAGES === "true";
    case "NEXT_PUBLIC_USE_SUPABASE_FILES":
      return process.env.NEXT_PUBLIC_USE_SUPABASE_FILES === "true";
    case "NEXT_PUBLIC_USE_SUPABASE_REALTIME":
      return process.env.NEXT_PUBLIC_USE_SUPABASE_REALTIME === "true";
    case "NEXT_PUBLIC_USE_SUPABASE_STORAGE":
      return process.env.NEXT_PUBLIC_USE_SUPABASE_STORAGE === "true";
    default:
      return false;
  }
}

export function useSupabaseThreads(): boolean {
  return isSupabaseFlagEnabled("NEXT_PUBLIC_USE_SUPABASE_THREADS") && !!getSupabaseClient();
}

export function useSupabaseMessages(): boolean {
  return isSupabaseFlagEnabled("NEXT_PUBLIC_USE_SUPABASE_MESSAGES") && !!getSupabaseClient();
}

export function useSupabaseFiles(): boolean {
  return isSupabaseFlagEnabled("NEXT_PUBLIC_USE_SUPABASE_FILES") && !!getSupabaseClient();
}

export function useSupabaseRealtime(): boolean {
  return isSupabaseFlagEnabled("NEXT_PUBLIC_USE_SUPABASE_REALTIME") && !!getSupabaseClient();
}

export function useSupabaseStorage(): boolean {
  return isSupabaseFlagEnabled("NEXT_PUBLIC_USE_SUPABASE_STORAGE") && !!getSupabaseClient();
}
