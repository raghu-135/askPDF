import type { BackendSentence, BBox } from './bbox-derivation';
import type {
  ProcessStatus as ProcessStatusValue,
  ThreadFileSourceType as ThreadFileSourceTypeValue,
} from './enums';
import type { WorkspaceTab, TraceWorkspaceTab, ResearchCanvasWorkspaceTab } from '../components/workbench/WorkspaceTabs';

type Sentence = Omit<BackendSentence, 'bboxes'> & { bboxes: BBox[] };
type DocumentProcessStatus = Extract<ProcessStatusValue, 'pending' | 'completed' | 'failed'>;

export type PdfTab = {
  id: string;
  fileName: string;
  fileHash: string;
  downloadUrl: string;
  sentences: Sentence[] | null;
  text?: string;
  sourceType?: ThreadFileSourceTypeValue;
  sourceUrl?: string;
  addedAt?: string;
  parsingStatus?: DocumentProcessStatus;
  associationScope?: 'thread' | 'project';
  isProjectKnowledge?: boolean;
  processingError?: string;
};

export type TraceTabStatusInput = {
  running?: boolean;
  error?: string | null;
};

export const PROJECT_OVERVIEW_TAB_ID = 'project-tab' as const;
export const THREAD_OVERVIEW_TAB_ID = 'thread-tab' as const;
export const EMBEDDINGS_TAB_ID = 'embeddings-tab' as const;
export const DOCUMENTS_TAB_ID = 'documents-tab' as const;

export const traceWorkspaceStatus = (
  traces: readonly TraceTabStatusInput[],
): TraceWorkspaceTab['status'] => {
  if (traces.some((trace) => Boolean(trace.error))) return 'failed';
  if (traces.some((trace) => Boolean(trace.running))) return 'running';
  return 'idle';
};

export const isBrowserWorkspaceActive = ({
  activeTabId,
}: {
  activeTabId: string | null;
  isBrowserActive?: boolean;
}) => activeTabId === 'browser-tab';

export const isDocumentsWorkspaceActive = (activeTabId: string | null) => (
  activeTabId === DOCUMENTS_TAB_ID
);

export const selectedWorkspaceTabValue = (
  tabs: readonly { id: string }[],
  activeTabId: string | null,
) => (activeTabId && tabs.some((tab) => tab.id === activeTabId) ? activeTabId : false);

export const buildDocumentWorkspaceTabs = ({
  enabled,
  documentCount = 0,
  traces,
  includeResearchCanvas = false,
  canvasCount = 0,
}: {
  enabled: boolean;
  documentCount?: number;
  traces: readonly TraceTabStatusInput[];
  includeResearchCanvas?: boolean;
  canvasCount?: number;
}): WorkspaceTab[] => {
  if (!enabled) return [];
  const tabs: WorkspaceTab[] = [
    { kind: 'thread', id: THREAD_OVERVIEW_TAB_ID, label: 'Thread' },
    { kind: 'memory', id: 'memory-tab', label: 'Memory' },
    { kind: 'documents', id: DOCUMENTS_TAB_ID, label: 'Documents', count: documentCount },
    { kind: 'browser', id: 'browser-tab', label: 'Browser' },
  ];
  if (includeResearchCanvas) {
    tabs.push({
      kind: 'research_canvas',
      id: 'research-canvas-tab',
      label: 'Canvas',
      count: canvasCount,
    } satisfies ResearchCanvasWorkspaceTab);
  }
  tabs.push({
    kind: 'embeddings',
    id: EMBEDDINGS_TAB_ID,
    label: 'Embeddings',
  });
  tabs.push({
    kind: 'trace',
    id: 'trace-tab',
    label: 'Debug Trace',
    count: traces.length,
    status: traceWorkspaceStatus(traces),
  });
  return tabs;
};

export const buildProjectWorkspaceTabs = (documentCount = 0): WorkspaceTab[] => [
  { kind: 'project', id: PROJECT_OVERVIEW_TAB_ID, label: 'Project' },
  { kind: 'memory', id: 'memory-tab', label: 'Memory' },
  { kind: 'documents', id: DOCUMENTS_TAB_ID, label: 'Documents', count: documentCount },
  { kind: 'browser', id: 'browser-tab', label: 'Browser' },
];

export const projectWorkspaceLandingTabId = (
  _documents: readonly Pick<PdfTab, 'id'>[] = [],
): string => PROJECT_OVERVIEW_TAB_ID;

export const threadWorkspaceLandingTabId = (
  documents: readonly Pick<PdfTab, 'id'>[] = [],
): string => (documents.length > 0 ? DOCUMENTS_TAB_ID : THREAD_OVERVIEW_TAB_ID);

export const buildHomeWorkspaceTabs = (): WorkspaceTab[] => [
  { kind: 'home', id: 'home-tab', label: 'Home' },
  { kind: 'memory', id: 'memory-tab', label: 'Memory' },
];
