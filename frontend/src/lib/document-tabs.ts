import type { BackendSentence, BBox } from './bbox-derivation';
import type {
  ProcessStatus as ProcessStatusValue,
  ThreadFileSourceType as ThreadFileSourceTypeValue,
} from './enums';
import type { WorkspaceTab, TraceWorkspaceTab } from '../components/workbench/WorkspaceTabs';

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
  parsingStatus?: DocumentProcessStatus;
  associationScope?: 'thread' | 'project';
  isProjectKnowledge?: boolean;
  processingError?: string;
};

export type TraceTabStatusInput = {
  running?: boolean;
  error?: string | null;
};

export const traceWorkspaceStatus = (
  traces: readonly TraceTabStatusInput[],
): TraceWorkspaceTab['status'] => {
  if (traces.some((trace) => Boolean(trace.error))) return 'failed';
  if (traces.some((trace) => Boolean(trace.running))) return 'running';
  return 'idle';
};

export const isBrowserWorkspaceActive = ({
  activeTabId,
  isBrowserActive,
}: {
  activeTabId: string | null;
  isBrowserActive: boolean;
}) => isBrowserActive || activeTabId === 'browser-tab';

export const buildDocumentWorkspaceTabs = ({
  enabled,
  documents,
  traces,
}: {
  enabled: boolean;
  documents: readonly PdfTab[];
  traces: readonly TraceTabStatusInput[];
}): WorkspaceTab[] => {
  if (!enabled) return [];
  return [
    { kind: 'browser', id: 'browser-tab', label: 'Browser' },
    { kind: 'memory', id: 'memory-tab', label: 'Memory' },
    ...documents.map((tab) => ({ ...tab, kind: 'document' as const })),
    {
      kind: 'trace',
      id: 'trace-tab',
      label: 'Debug Trace',
      count: traces.length,
      status: traceWorkspaceStatus(traces),
    },
  ];
};

export const buildProjectWorkspaceTabs = (documents: readonly PdfTab[]): WorkspaceTab[] => [
  { kind: 'browser', id: 'browser-tab', label: 'Browser' },
  { kind: 'memory', id: 'memory-tab', label: 'Memory' },
  ...documents.map((tab) => ({ ...tab, kind: 'document' as const })),
];

export const buildHomeWorkspaceTabs = (): WorkspaceTab[] => [
  { kind: 'home', id: 'home-tab', label: 'Home' },
  { kind: 'memory', id: 'memory-tab', label: 'Memory' },
];
