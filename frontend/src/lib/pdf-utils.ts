import type { FormattedSelection } from "@embedpdf/plugin-selection/react";
import type { PdfTab } from "../components/PdfTabs";

export type PdfSelectionBBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type SentenceWithPageBBoxes<TBBox extends PdfSelectionBBox = PdfSelectionBBox> = {
  pageBBoxes: TBBox[];
};

/**
 * Truncates a file name for display, preserving the extension if possible.
 * @param name The file name to truncate.
 * @param maxLen The maximum length of the truncated name (default: 20).
 * @returns The truncated file name, preserving the extension if possible.
 */
export function truncateFileName(name: string, maxLen: number = 20): string {
  if (name.length <= maxLen) return name;
  const ext = name.lastIndexOf('.');
  if (ext > 0 && name.length - ext <= 5) {
    // Preserve extension
    const extStr = name.substring(ext);
    const baseName = name.substring(0, ext);
    const availableLen = maxLen - extStr.length - 3;
    if (availableLen > 5) {
      return baseName.substring(0, availableLen) + '...' + extStr;
    }
  }
  return name.substring(0, maxLen - 3) + '...';
}

/**
 * Handles switching the active PDF tab and resets related state.
 *
 * @param tabId The ID of the tab to activate.
 * @param setActiveTabId Function to set the active tab ID.
 * @param setCurrentPdfId Function to set the current PDF ID (reset to null).
 * @param setPlayRequestId Function to set the play request ID (reset to null).
 * @param setActiveSource Function to set the active source (set to 'pdf').
 */
export function handleTabChangeUtil(
  tabId: string,
  setActiveTabId: (id: string | null) => void,
  setCurrentPdfId: (id: number | null) => void,
  setPlayRequestId: (id: number | null) => void,
  setActiveSource: (src: 'pdf' | 'chat') => void
) {
  setActiveTabId(tabId);
  setCurrentPdfId(null);
  setPlayRequestId(null);
  setActiveSource('pdf');
}

/**
 * Handles closing a PDF tab and updates state accordingly.
 *
 * @param tabId The ID of the tab to close.
 * @param pdfTabs The current list of PDF tabs.
 * @param activeTabId The currently active tab ID.
 * @param setPdfTabs Function to update the list of PDF tabs.
 * @param setActiveTabId Function to set the active tab ID.
 * @param setCurrentPdfId Function to set the current PDF ID (reset to null).
 * @param setPlayRequestId Function to set the play request ID (reset to null).
 */
export function handleTabCloseUtil(
  tabId: string,
  pdfTabs: PdfTab[],
  activeTabId: string | null,
  setPdfTabs: (tabs: PdfTab[]) => void,
  setActiveTabId: (id: string | null) => void,
  setCurrentPdfId: (id: number | null) => void,
  setPlayRequestId: (id: number | null) => void
) {
  const newTabs = pdfTabs.filter(t => t.id !== tabId);
  // If closing the active tab, switch to another tab
  if (activeTabId === tabId) {
    const closingIndex = pdfTabs.findIndex(t => t.id === tabId);
    if (newTabs.length > 0) {
      // Try to select the tab to the left, or the first tab
      const newIndex = Math.max(0, closingIndex - 1);
      setActiveTabId(newTabs[newIndex]?.id || null);
    } else {
      setActiveTabId(null);
    }
  }
  setPdfTabs(newTabs);
  setCurrentPdfId(null);
  setPlayRequestId(null);
}

/**
 * Returns the active tab from the list by its ID.
 *
 * @param pdfTabs The list of PDF tabs.
 * @param activeTabId The ID of the active tab.
 * @returns The active PdfTab object, or null if not found.
 */
export function getActiveTab(pdfTabs: PdfTab[], activeTabId: string | null): PdfTab | null {
  return pdfTabs.find(t => t.id === activeTabId) || null;
}

/**
 * Returns the sentences, URL, file hash, and file name for the active tab.
 *
 * @param activeTab The currently active PdfTab, or null.
 * @returns An object containing pdfSentences, pdfUrl, fileHash, and fileName.
 */
export function getActiveTabData(activeTab: PdfTab | null): {
  pdfSentences: any[];
  pdfUrl: string | null;
  fileHash: string | null;
  fileName: string | null;
} {
  return {
    pdfSentences: activeTab?.sentences || [],
    pdfUrl: activeTab?.pdfUrl || null,
    fileHash: activeTab?.fileHash || null,
    fileName: activeTab?.fileName || null,
  };
}

function rectOverlapArea(
  a: { left: number; top: number; right: number; bottom: number },
  b: { left: number; top: number; right: number; bottom: number },
) {
  const width = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
  const height = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
  return width * height;
}

function bboxToRect(bbox: PdfSelectionBBox) {
  return {
    left: bbox.x,
    top: bbox.y,
    right: bbox.x + bbox.width,
    bottom: bbox.y + bbox.height,
  };
}

export function findSentenceForSelection<TSentence extends SentenceWithPageBBoxes>(
  selections: FormattedSelection[],
  byPage: { [key: number]: TSentence[] },
): TSentence | null {
  let best: { sentence: TSentence; score: number } | null = null;

  for (const selection of selections) {
    const pageData = byPage[selection.pageIndex + 1] || [];
    const selectionRect = {
      left: selection.rect.origin.x,
      top: selection.rect.origin.y,
      right: selection.rect.origin.x + selection.rect.size.width,
      bottom: selection.rect.origin.y + selection.rect.size.height,
    };

    for (const sentence of pageData) {
      for (const bbox of sentence.pageBBoxes) {
        const score = rectOverlapArea(selectionRect, bboxToRect(bbox));
        if (!best || score > best.score) {
          best = { sentence, score };
        }
      }
    }
  }

  if (best && best.score > 0) return best.sentence;

  for (const selection of selections) {
    const pageData = byPage[selection.pageIndex + 1] || [];
    const center = {
      x: selection.rect.origin.x + selection.rect.size.width / 2,
      y: selection.rect.origin.y + selection.rect.size.height / 2,
    };

    for (const sentence of pageData) {
      for (const bbox of sentence.pageBBoxes) {
        const rect = bboxToRect(bbox);
        if (
          center.x >= rect.left &&
          center.x <= rect.right &&
          center.y >= rect.top &&
          center.y <= rect.bottom
        ) {
          return sentence;
        }
      }
    }
  }

  return null;
}
