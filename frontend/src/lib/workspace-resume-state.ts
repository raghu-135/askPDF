import {
  DOCUMENTS_TAB_ID,
  PROJECT_OVERVIEW_TAB_ID,
  THREAD_OVERVIEW_TAB_ID,
} from './document-tabs.ts';
import { RESEARCH_CANVAS_TAB_ID } from './canvas-spec.ts';

export const WORKSPACE_RESUME_STORAGE_KEY = 'askpdf.workspace.resume.v1';

export type WorkspaceResumeState = {
  tabId: string;
  documentId?: string | null;
  traceId?: string | null;
  canvasId?: string | null;
};

type WorkspaceResumeMap = Record<string, WorkspaceResumeState>;

const MEMORY_TAB_ID = 'memory-tab';
const HOME_TAB_ID = 'home-tab';
const TRACE_TAB_ID = 'trace-tab';

const normalizeResumeState = (value: unknown): WorkspaceResumeState | null => {
  if (!value || typeof value !== 'object') return null;
  const record = value as Record<string, unknown>;
  if (typeof record.tabId !== 'string' || !record.tabId.trim()) return null;
  return {
    tabId: record.tabId,
    documentId: typeof record.documentId === 'string' ? record.documentId : record.documentId === null ? null : undefined,
    traceId: typeof record.traceId === 'string' ? record.traceId : record.traceId === null ? null : undefined,
    canvasId: typeof record.canvasId === 'string' ? record.canvasId : record.canvasId === null ? null : undefined,
  };
};

const readResumeMap = (): WorkspaceResumeMap => {
  if (typeof window === 'undefined') return {};
  try {
    const raw = window.localStorage.getItem(WORKSPACE_RESUME_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return {};
    const map: WorkspaceResumeMap = {};
    for (const [key, value] of Object.entries(parsed)) {
      const normalized = normalizeResumeState(value);
      if (normalized) map[key] = normalized;
    }
    return map;
  } catch {
    return {};
  }
};

export const defaultOverviewTabId = (contextKey: string): string => {
  if (contextKey.startsWith('thread:')) return THREAD_OVERVIEW_TAB_ID;
  if (contextKey.startsWith('project:')) return PROJECT_OVERVIEW_TAB_ID;
  return HOME_TAB_ID;
};

export const readWorkspaceResume = (contextKey: string): WorkspaceResumeState | null => {
  const map = readResumeMap();
  return map[contextKey] ?? null;
};

const resumeStatesEqual = (
  left: WorkspaceResumeState | null | undefined,
  right: WorkspaceResumeState,
): boolean => (
  left?.tabId === right.tabId
  && (left?.documentId ?? null) === (right.documentId ?? null)
  && (left?.traceId ?? null) === (right.traceId ?? null)
  && (left?.canvasId ?? null) === (right.canvasId ?? null)
);

export const writeWorkspaceResume = (
  contextKey: string,
  state: WorkspaceResumeState,
): void => {
  if (typeof window === 'undefined') return;
  if (!state.tabId || state.tabId === MEMORY_TAB_ID) return;
  try {
    const map = readResumeMap();
    const nextState = {
      tabId: state.tabId,
      documentId: state.documentId ?? null,
      traceId: state.traceId ?? null,
      canvasId: state.canvasId ?? null,
    };
    if (resumeStatesEqual(map[contextKey], nextState)) return;
    map[contextKey] = nextState;
    window.localStorage.setItem(WORKSPACE_RESUME_STORAGE_KEY, JSON.stringify(map));
  } catch {
    // Persistence is best effort.
  }
};

export const resolveWorkspaceResume = ({
  contextKey,
  state,
  availableTabs,
  pdfTabs = [],
  traceIds = [],
  canvasIds = [],
}: {
  contextKey: string;
  state?: WorkspaceResumeState | null;
  availableTabs: readonly { id: string }[];
  pdfTabs?: readonly { id: string }[];
  traceIds?: readonly string[];
  canvasIds?: readonly string[];
}): WorkspaceResumeState => {
  const fallbackTabId = defaultOverviewTabId(contextKey);
  const availableTabIds = new Set(availableTabs.map((tab) => tab.id));
  const emptyState = {
    tabId: fallbackTabId,
    documentId: null,
    traceId: null,
    canvasId: null,
  };

  if (!state || state.tabId === MEMORY_TAB_ID || !availableTabIds.has(state.tabId)) {
    return emptyState;
  }

  const resolved: WorkspaceResumeState = {
    tabId: state.tabId,
    documentId: state.documentId ?? null,
    traceId: state.traceId ?? null,
    canvasId: state.canvasId ?? null,
  };

  if (resolved.tabId === DOCUMENTS_TAB_ID) {
    if (pdfTabs.length === 0) {
      return emptyState;
    }
    const documentExists = resolved.documentId
      ? pdfTabs.some((tab) => tab.id === resolved.documentId)
      : false;
    resolved.documentId = documentExists
      ? resolved.documentId
      : pdfTabs[0]?.id ?? null;
    return resolved;
  }

  if (resolved.tabId === TRACE_TAB_ID) {
    if (resolved.traceId && traceIds.includes(resolved.traceId)) {
      return resolved;
    }
    resolved.traceId = traceIds[0] ?? null;
    return resolved;
  }

  if (resolved.tabId === RESEARCH_CANVAS_TAB_ID) {
    if (resolved.canvasId && canvasIds.includes(resolved.canvasId)) {
      return resolved;
    }
    resolved.canvasId = canvasIds[0] ?? null;
    return resolved;
  }

  resolved.documentId = null;
  resolved.traceId = null;
  resolved.canvasId = null;
  return resolved;
};
