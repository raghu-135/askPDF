export type IdentifiedInstanceTab = { id: string };

export const selectedInstanceTabValue = (
  tabs: readonly IdentifiedInstanceTab[],
  activeId: string | null,
): string | false => (
  activeId && tabs.some((tab) => tab.id === activeId) ? activeId : false
);

export const closeInstanceTab = <T extends IdentifiedInstanceTab>(
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

export const mergeInstanceTab = <T extends IdentifiedInstanceTab>(
  tab: T | undefined,
  nextTab: T,
): T => (tab ? { ...tab, ...nextTab } : nextTab);

export const upsertInstanceTab = <T extends IdentifiedInstanceTab>(
  tabs: T[],
  nextTab: T,
  options?: {
    canOpen?: (tab: T) => boolean;
    merge?: (existing: T | undefined, incoming: T) => T;
    replaceIds?: (existingId: string, incoming: T) => string[];
  },
): T[] => {
  const canOpen = options?.canOpen ?? (() => Boolean(nextTab?.id));
  if (!canOpen(nextTab)) return tabs;

  let working = tabs;
  if (options?.replaceIds) {
    const replaceIds = new Set(options.replaceIds('', nextTab));
    working = tabs.filter((tab) => !replaceIds.has(tab.id));
  }

  const index = working.findIndex((tab) => tab.id === nextTab.id);
  const merge = options?.merge ?? mergeInstanceTab;
  if (index < 0) return [...working, nextTab];
  return working.map((tab, currentIndex) => (
    currentIndex === index ? merge(tab, nextTab) : tab
  ));
};