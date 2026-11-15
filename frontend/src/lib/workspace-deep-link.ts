import { RESEARCH_CANVAS_TAB_ID } from './canvas-spec.ts';

export const WORKSPACE_DEEP_LINK_PARAMS = {
  project: 'project',
  thread: 'thread',
  canvas: 'canvas',
} as const;

export type CanvasWorkspaceDeepLink = {
  projectId?: string | null;
  threadId: string;
  canvasId: string;
};

export const buildCanvasWorkspaceDeepLink = ({
  threadId,
  projectId,
  canvasId,
  origin = typeof window !== 'undefined' ? window.location.origin : 'http://localhost:3000',
  pathname = typeof window !== 'undefined' ? window.location.pathname : '/',
}: {
  threadId: string;
  canvasId: string;
  projectId?: string | null;
  origin?: string;
  pathname?: string;
}): string => {
  const url = new URL(pathname, origin);
  url.search = '';
  url.searchParams.set(WORKSPACE_DEEP_LINK_PARAMS.thread, threadId);
  url.searchParams.set(WORKSPACE_DEEP_LINK_PARAMS.canvas, canvasId);
  if (projectId) {
    url.searchParams.set(WORKSPACE_DEEP_LINK_PARAMS.project, projectId);
  }
  return url.toString();
};

export const parseCanvasWorkspaceDeepLink = (
  search: string,
): CanvasWorkspaceDeepLink | null => {
  const params = new URLSearchParams(search.startsWith('?') ? search.slice(1) : search);
  const threadId = params.get(WORKSPACE_DEEP_LINK_PARAMS.thread)?.trim();
  const canvasId = params.get(WORKSPACE_DEEP_LINK_PARAMS.canvas)?.trim();
  if (!threadId || !canvasId) return null;
  const projectId = params.get(WORKSPACE_DEEP_LINK_PARAMS.project)?.trim() || null;
  return { threadId, canvasId, projectId };
};

export const canvasDeepLinkResumeState = (canvasId: string) => ({
  tabId: RESEARCH_CANVAS_TAB_ID,
  documentId: null,
  traceId: null,
  canvasId,
});

export const stripCanvasWorkspaceDeepLink = (): void => {
  if (typeof window === 'undefined') return;
  const url = new URL(window.location.href);
  let changed = false;
  for (const key of Object.values(WORKSPACE_DEEP_LINK_PARAMS)) {
    if (url.searchParams.has(key)) {
      url.searchParams.delete(key);
      changed = true;
    }
  }
  if (!changed) return;
  const next = `${url.pathname}${url.search}${url.hash}`;
  window.history.replaceState(window.history.state, '', next);
};