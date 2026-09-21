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
  /** When false, open and highlight the sentence without starting TTS. */
  play?: boolean;
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

export const DAG_NODE_WIDTH = 148;
export const DAG_NODE_HEIGHT = 40;
export const DAG_LAYER_GAP = 56;
export const DAG_ROW_GAP = 20;

export type CanvasDagLayoutNode = {
  id: string;
  label: string;
  x: number;
  y: number;
};

export type CanvasDagLayoutEdge = {
  source: string;
  target: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type CanvasDagLayout = {
  width: number;
  height: number;
  nodes: CanvasDagLayoutNode[];
  edges: CanvasDagLayoutEdge[];
};

export const layoutCanvasDag = (
  nodes: Array<{ id: string; label: string }>,
  edges: Array<{ source: string; target: string }>,
): CanvasDagLayout => {
  const incoming = new Map(nodes.map((node) => [node.id, 0]));
  const outgoing = new Map(nodes.map((node) => [node.id, [] as string[]]));
  for (const edge of edges) {
    if (!incoming.has(edge.source) || !incoming.has(edge.target)) continue;
    incoming.set(edge.target, (incoming.get(edge.target) || 0) + 1);
    outgoing.get(edge.source)?.push(edge.target);
  }

  const remaining = new Set(nodes.map((node) => node.id));
  const layers: string[][] = [];
  while (remaining.size) {
    const ready = [...remaining].filter((id) => (incoming.get(id) || 0) === 0);
    const layer = ready.length ? ready : [[...remaining][0]];
    layers.push(layer);
    for (const id of layer) {
      remaining.delete(id);
      for (const next of outgoing.get(id) || []) {
        if (remaining.has(next)) incoming.set(next, Math.max(0, (incoming.get(next) || 0) - 1));
      }
    }
  }

  const byId = new Map(nodes.map((node) => [node.id, node]));
  const placed: CanvasDagLayoutNode[] = [];
  const positions = new Map<string, { x: number; y: number }>();
  layers.forEach((layer, column) => {
    layer.forEach((id, row) => {
      const node = byId.get(id);
      if (!node) return;
      const x = column * (DAG_NODE_WIDTH + DAG_LAYER_GAP);
      const y = row * (DAG_NODE_HEIGHT + DAG_ROW_GAP);
      placed.push({ id, label: node.label, x, y });
      positions.set(id, { x, y });
    });
  });

  const laidEdges: CanvasDagLayoutEdge[] = edges.flatMap((edge) => {
    const from = positions.get(edge.source);
    const to = positions.get(edge.target);
    if (!from || !to) return [];
    return [{
      source: edge.source,
      target: edge.target,
      x1: from.x + DAG_NODE_WIDTH,
      y1: from.y + DAG_NODE_HEIGHT / 2,
      x2: to.x,
      y2: to.y + DAG_NODE_HEIGHT / 2,
    }];
  });

  return {
    width: Math.max(DAG_NODE_WIDTH, layers.length * DAG_NODE_WIDTH + Math.max(0, layers.length - 1) * DAG_LAYER_GAP),
    height: Math.max(
      DAG_NODE_HEIGHT,
      Math.max(0, ...layers.map((layer) => layer.length)) * DAG_NODE_HEIGHT
        + Math.max(0, Math.max(0, ...layers.map((layer) => layer.length)) - 1) * DAG_ROW_GAP,
    ),
    nodes: placed,
    edges: laidEdges,
  };
};

export const canvasSpecCounts = (spec: CanvasSpec) => {
  const blocks = spec.sections.flatMap((section) => section.blocks);
  return {
    sections: spec.sections.length,
    stats: blocks.filter((block) => block.type === 'stat').length,
    sources: blocks.reduce((count, block) => (
      block.type === 'sources' ? count + block.citations.length : count
    ), 0),
  };
};
