import type { ReactNode } from 'react';

export const WORKSPACE_CHROME_SEPARATOR = Symbol('workspace-chrome-separator');

export type WorkspaceChromeEntry = ReactNode | typeof WORKSPACE_CHROME_SEPARATOR;

export function isWorkspaceChromeSeparator(
  entry: WorkspaceChromeEntry,
): entry is typeof WORKSPACE_CHROME_SEPARATOR {
  return entry === WORKSPACE_CHROME_SEPARATOR;
}

export function splitWorkspaceChromeRows(entries: readonly WorkspaceChromeEntry[]): ReactNode[][] {
  const rows: ReactNode[][] = [[]];

  for (const entry of entries) {
    if (isWorkspaceChromeSeparator(entry)) {
      rows.push([]);
      continue;
    }
    if (entry == null || entry === false) continue;
    rows[rows.length - 1].push(entry);
  }

  return rows.filter((row) => row.length > 0);
}

export function flattenWorkspaceChromeEntries(entries: readonly WorkspaceChromeEntry[]): ReactNode[] {
  return entries.filter((entry) => !isWorkspaceChromeSeparator(entry) && entry != null && entry !== false) as ReactNode[];
}
