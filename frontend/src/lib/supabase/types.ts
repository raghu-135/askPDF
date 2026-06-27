export interface SupabaseThreadRow {
  id: string;
  name: string;
  embed_model: string;
  settings: Record<string, any>;
  created_at: string;
  updated_at?: string | null;
  message_count?: number;
  file_count?: number;
  last_message_at?: string | null;
  latest_activity_at?: string | null;
}

export interface SupabaseFileRow {
  file_hash: string;
  file_name: string;
  file_path?: string | null;
  source_type?: "pdf" | "web" | string;
  file_status?: Record<string, any>;
  parsed_sentences_json?: string | null;
  storage_bucket?: string | null;
  storage_path?: string | null;
  created_at?: string;
  added_at?: string;
}

export interface SupabaseMessageRow {
  id: string;
  thread_id: string;
  role: "user" | "assistant";
  content: string;
  context_compact?: string | null;
  reasoning?: string | null;
  reasoning_available?: boolean;
  reasoning_format?: "structured" | "tagged_text" | "none" | string;
  web_sources?: any[] | null;
  created_at: string;
}

export interface SupabaseAnnotationRow {
  thread_id: string;
  file_hash: string;
  annotations_json: string;
  created_at?: string;
  updated_at?: string;
}
