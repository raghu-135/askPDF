export const RESEARCH_CANVAS_TAB_ID = 'research-canvas-tab';

export type CanvasCitationKind = 'document' | 'web' | 'memory' | 'conversation';

export type CanvasCitation = {
  kind: CanvasCitationKind;
  label: string;
  file_hash?: string | null;
  sentence_id?: number | null;
  url?: string | null;
  memory_id?: string | null;
  message_id?: string | null;
};

export type CanvasStatBlock = {
  type: 'stat';
  value: string;
  label: string;
  tone?: 'neutral' | 'info' | 'warning';
};

export type CanvasTableBlock = {
  type: 'table';
  caption?: string | null;
  headers: string[];
  rows: string[][];
};

export type CanvasCalloutBlock = {
  type: 'callout';
  tone: 'info' | 'warning';
  title: string;
  body: string;
};

export type CanvasMarkdownBlock = {
  type: 'markdown';
  text: string;
};

export type CanvasSourcesBlock = {
  type: 'sources';
  title?: string;
  citations: CanvasCitation[];
};

export type CanvasDagBlock = {
  type: 'dag';
  title?: string | null;
  nodes: Array<{ id: string; label: string }>;
  edges: Array<{ source: string; target: string }>;
};

export type CanvasBlock =
  | CanvasStatBlock
  | CanvasTableBlock
  | CanvasCalloutBlock
  | CanvasMarkdownBlock
  | CanvasSourcesBlock
  | CanvasDagBlock;

export type CanvasSection = {
  title: string;
  blocks: CanvasBlock[];
};

export type CanvasSpec = {
  schema_version: 1;
  title: string;
  summary?: string | null;
  sections: CanvasSection[];
};

export type ThreadCanvasRecord = {
  id: string;
  thread_id: string;
  chat_turn_id?: string | null;
  title: string;
  spec: CanvasSpec;
  supersedes_id?: string | null;
  created_at: string;
  current: boolean;
};

export type CanvasRef = {
  id: string;
  title: string;
};

export type DocumentCanvasCitationTarget = {
  fileHash: string;
  sentenceId: number | null;
};

export const documentCitationTarget = (
  citation: CanvasCitation,
): DocumentCanvasCitationTarget | null => {
  if (citation.kind !== 'document' || !citation.file_hash) return null;
  return {
    fileHash: citation.file_hash,
    sentenceId: Number.isInteger(citation.sentence_id) ? Number(citation.sentence_id) : null,
  };
};
