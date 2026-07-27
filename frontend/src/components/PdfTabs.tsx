import React from 'react';
import type { BackendSentence, BBox } from '../lib/bbox-derivation';
import { ProcessStatus, type ProcessStatus as ProcessStatusValue, type ThreadFileSourceType as ThreadFileSourceTypeValue } from '../lib/enums';
import WorkspaceTabs, { type WorkspaceTab } from './workbench/WorkspaceTabs';

type Sentence = Omit<BackendSentence, 'bboxes'> & { bboxes: BBox[] };

export type PdfTab = {
  id: string;
  fileName: string;
  fileHash: string;
  downloadUrl: string;
  sentences: Sentence[] | null;
  text?: string;
  sourceType?: ThreadFileSourceTypeValue;
  sourceUrl?: string;
  parsingStatus?: Extract<ProcessStatusValue, typeof ProcessStatus.Pending | typeof ProcessStatus.Completed | typeof ProcessStatus.Failed>;
};

type Props = {
  tabs: PdfTab[];
  activeTabId: string | null;
  onTabChange: (tabId: string) => void;
  onTabClose: (tabId: string) => void;
  /** When provided, shows a remove-from-thread trash icon on each tab. */
  onTabRemove?: (tabId: string) => void;
  darkMode?: boolean;
  /** Whether to show the browser tab */
  showBrowserTab?: boolean;
  /** Callback when browser tab is clicked */
  onBrowserTabClick?: () => void;
  /** Callback when add-to-thread button is clicked on browser tab */
  onAddBrowserToThread?: () => void;
  /** Whether browser capture is in progress */
  isBrowserCapturing?: boolean;
};

const PdfTabs = React.memo(function PdfTabs({ tabs, activeTabId, onTabChange, onTabClose, onTabRemove, showBrowserTab = false, onBrowserTabClick, onAddBrowserToThread, isBrowserCapturing = false }: Props) {
  const browserTabId = 'browser-tab' as const;

  if (tabs.length === 0 && !showBrowserTab) {
    return null;
  }

  const workspaceTabs: WorkspaceTab[] = [
    ...(showBrowserTab ? [{ kind: 'browser' as const, id: browserTabId, label: 'Browser' }] : []),
    ...tabs.map((tab) => ({ ...tab, kind: 'document' as const })),
  ];

  return (
    <WorkspaceTabs
      tabs={workspaceTabs}
      activeTabId={activeTabId}
      onTabChange={(tabId) => {
        if (tabId === browserTabId) {
          onBrowserTabClick?.();
          return;
        }
        onTabChange(tabId);
      }}
      onTabClose={onTabClose}
      onDocumentRemove={onTabRemove}
      onAddBrowserToThread={onAddBrowserToThread}
      isBrowserCapturing={isBrowserCapturing}
    />
  );
});

export default PdfTabs;
