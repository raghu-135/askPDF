export type IdentifiedTraceTab = { id: string; liveTraceView?: unknown };

export const upsertTraceTab = <T extends IdentifiedTraceTab>(tabs: T[], nextTab: T): T[] => {
  const index = tabs.findIndex((tab) => tab.id === nextTab.id);
  if (index < 0) return [...tabs, nextTab];
  return tabs.map((tab, currentIndex) => {
    if (currentIndex !== index) return tab;
    const liveTraceView = nextTab.liveTraceView ?? tab.liveTraceView;
    return {
      ...tab,
      ...nextTab,
      ...(liveTraceView === undefined ? {} : { liveTraceView }),
    };
  });
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
