import {
  closeInstanceTab,
  mergeInstanceTab,
  upsertInstanceTab,
  type IdentifiedInstanceTab,
} from './instance-tabs.ts';

export type IdentifiedTraceTab = IdentifiedInstanceTab & { messageId?: string; liveTraceView?: unknown };

export const PENDING_TRACE_PREFIX = 'pending-trace:';

export const isPlaceholderTraceId = (value: unknown): boolean => (
  typeof value === 'string'
  && (value.startsWith('temp-assistant-') || value.startsWith('test-assistant-'))
);

export const isPendingTraceId = (value: unknown): boolean => (
  typeof value === 'string' && value.startsWith(PENDING_TRACE_PREFIX)
);

export const isValidTraceId = (value: unknown): value is string => (
  typeof value === 'string' && value.trim().length > 0 && !isPlaceholderTraceId(value) && !isPendingTraceId(value)
);

export const pendingTraceTabId = (messageId: string) => `${PENDING_TRACE_PREFIX}${messageId}`;

export const canOpenTraceTab = (value: unknown): value is string => (
  isValidTraceId(value) || isPendingTraceId(value)
);

export const traceTabIdForRun = (runId: string | null | undefined, messageId: string) => (
  isValidTraceId(runId) ? runId : pendingTraceTabId(messageId)
);

export const mergeTraceTab = <T extends IdentifiedTraceTab>(tab: T | undefined, nextTab: T): T => {
  if (!tab) return nextTab;
  const liveTraceView = nextTab.liveTraceView ?? tab.liveTraceView;
  return mergeInstanceTab(tab, {
    ...nextTab,
    ...(liveTraceView === undefined ? {} : { liveTraceView }),
  });
};

export const upsertTraceTab = <T extends IdentifiedTraceTab>(tabs: T[], nextTab: T): T[] => {
  const replaceIds = isValidTraceId(nextTab.id) && nextTab.messageId
    ? [pendingTraceTabId(nextTab.messageId as string)]
    : [];
  return upsertInstanceTab(tabs, nextTab, {
    canOpen: (tab) => canOpenTraceTab(tab.id),
    merge: mergeTraceTab,
    replaceIds: () => replaceIds,
  });
};

export const closeTraceTab = closeInstanceTab;
