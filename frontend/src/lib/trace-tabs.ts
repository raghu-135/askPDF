export type IdentifiedTraceTab = { id: string; messageId?: string; liveTraceView?: unknown };

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
  return {
    ...tab,
    ...nextTab,
    ...(liveTraceView === undefined ? {} : { liveTraceView }),
  };
};

export const upsertTraceTab = <T extends IdentifiedTraceTab>(tabs: T[], nextTab: T): T[] => {
  if (!canOpenTraceTab(nextTab?.id)) return tabs;
  const withoutPending = isValidTraceId(nextTab.id) && nextTab.messageId
    ? tabs.filter((tab) => tab.id !== pendingTraceTabId(nextTab.messageId as string))
    : tabs;
  const index = withoutPending.findIndex((tab) => tab.id === nextTab.id);
  if (index < 0) return [...withoutPending, nextTab];
  return withoutPending.map((tab, currentIndex) => currentIndex === index ? mergeTraceTab(tab, nextTab) : tab);
};

export const closeTraceTab = <T extends IdentifiedTraceTab>(
  tabs: T[],
  activeId: string | null,
  closingId: string,
): { tabs: T[]; activeId: string | null } => {
  const closingIndex = tabs.findIndex((tab) => tab.id === closingId);
  const nextTabs = tabs.filter((tab) => tab.id !== closingId);
  if (activeId !== closingId) return { tabs: nextTabs, activeId };
  return {
    tabs: nextTabs,
    activeId: nextTabs[Math.max(0, closingIndex - 1)]?.id || nextTabs[0]?.id || null,
  };
};
