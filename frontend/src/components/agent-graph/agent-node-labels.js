const FALLBACK_NODE_LABELS = {
  context_loader: 'Context Loader',
  router: 'Router',
  planner: 'Planner',
  evidence_evaluator: 'Evidence Evaluator',
  replanner: 'Replanner',
  retrieval_worker: 'Document Retrieval',
  thread_conversation_history_worker: 'Thread Conversation History',
  durable_memory_worker: 'Durable Memory',
  thread_events_worker: 'Thread Events Retrieval',
  web_approval_gate: 'Web Approval',
  web_worker: 'Web Retrieval',
  direct_answer: 'Direct Answer',
  synthesizer: 'Synthesizer',
  finalizer: 'Finalizer',
  hitl_gate: 'HITL Gate',
};

const titleizeNodeId = (id) => (
  id.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase())
);

const catalogDisplayName = (catalog, type) => {
  if (!catalog || !type) return undefined;
  const entry = catalog[type];
  if (!entry) return undefined;
  if (typeof entry.displayName === 'string' && entry.displayName) return entry.displayName;
  if (typeof entry.display_name === 'string' && entry.display_name) return entry.display_name;
  return undefined;
};

export const formatNodeLabel = (id, type, catalog) => (
  FALLBACK_NODE_LABELS[id]
  || catalogDisplayName(catalog, type)
  || FALLBACK_NODE_LABELS[type || '']
  || titleizeNodeId(id)
);

export const formatNodeInstanceLabel = (id, type) => (
  type && id !== type ? `${id} · ${type}` : id
);
