export type IdentifiedTraceTab = { id: string };

export const upsertTraceTab = <T extends IdentifiedTraceTab>(tabs: T[], nextTab: T): T[] => {
  const index = tabs.findIndex((tab) => tab.id === nextTab.id);
  if (index < 0) return [...tabs, nextTab];
  return tabs.map((tab, currentIndex) => currentIndex === index ? { ...tab, ...nextTab } : tab);
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
