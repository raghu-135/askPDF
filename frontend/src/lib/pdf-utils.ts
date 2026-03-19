import { PdfTab } from "../components/PdfTabs";
import { DocumentSource } from "./api";

/**
 * Clears any active source highlights in the UI.
 */
export function clearSourceHighlightsUtil(
  setHighlightedSourceSentenceIds: (ids: number[]) => void,
  setHighlightedSourceFileHash: (hash: string | null) => void,
  setHighlightedSourceMessageId: (id: string | null) => void
) {
  setHighlightedSourceSentenceIds([]);
  setHighlightedSourceFileHash(null);
  setHighlightedSourceMessageId(null);
}

/**
 * High-level orchestration for highlighting sources from a chat message. 
 * Handles tab switching, ID aggregation, and focus.
 */
export function handleHighlightSourcesUtil(
  payload: { messageId: string; documentSources: DocumentSource[]; matchedSentenceIds?: number[] },
  highlightedSourceMessageId: string | null,
  pdfTabs: PdfTab[],
  activeTab: PdfTab | null,
  activeTabId: string | null,
  setPdfTabs: (tabs: PdfTab[]) => void,
  setActiveTabId: (id: string | null) => void,
  setHighlightedSourceSentenceIds: (ids: number[]) => void,
  setHighlightedSourceFileHash: (hash: string | null) => void,
  setHighlightedSourceMessageId: (id: string | null) => void,
  setActiveSource: (src: 'pdf' | 'chat') => void,
  setCurrentPdfId: (id: number | null) => void,
  mode: 'all' | 'focused' = 'all',
  threshold: number = 1.0
) {
  const pdfSources = (payload.documentSources || []).filter(s => (s.source_type ?? 'pdf') === 'pdf');
  if (pdfSources.length === 0) return;

  const activeFileHash = activeTab?.fileHash;
  const preferredFileHash = activeFileHash && pdfSources.some(s => s.file_hash === activeFileHash)
    ? activeFileHash
    : pdfSources[0].file_hash;

  const targetTab = pdfTabs.find(t => t.fileHash === preferredFileHash);
  if (!targetTab) return;

  if (activeTabId !== targetTab.id) {
    setActiveTabId(targetTab.id);
  }

  const targetSources = pdfSources.filter(s => s.file_hash === preferredFileHash);

  // Figure out which IDs to show based on mode
  let matchedIds: number[] = [];

  if (mode === 'focused') {
    // Strategy: show top X% of results by count.
    const validTargetSources = targetSources.filter(s => s.score !== undefined && s.score !== null && s.score > 0);
    const baseSources = validTargetSources.length > 0 ? validTargetSources : targetSources;
    
    // targetSources are already sorted by score descending from the backend.
    const ratio = Math.max(0.05, Math.min(1.0, threshold)); // Clamp ratio between 5% and 100%
    const count = Math.max(1, Math.ceil(baseSources.length * ratio));
    const focusedSources = [...baseSources]
      .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
      .slice(0, count);

    const idSet = new Set<number>();
    focusedSources.forEach(s => {
      if (s.sentence_ids) s.sentence_ids.forEach(id => idSet.add(Number(id)));
    });
    matchedIds = Array.from(idSet);
  } else {
    // Show all matching IDs but filter out context expansion pollution (score 0.0)
    const validTargetSources = targetSources.filter(s => s.score !== undefined && s.score !== null && s.score > 0);
    const baseSources = validTargetSources.length > 0 ? validTargetSources : targetSources;
    
    const idSet = new Set<number>();
    baseSources.forEach(s => {
      if (s.sentence_ids) s.sentence_ids.forEach(id => idSet.add(Number(id)));
    });
    matchedIds = Array.from(idSet);
  }

  setHighlightedSourceSentenceIds(matchedIds);
  setHighlightedSourceFileHash(preferredFileHash);
  setHighlightedSourceMessageId(payload.messageId);
  setActiveSource('pdf');
  if (matchedIds.length > 0) {
    setCurrentPdfId(matchedIds[0]);
  }
}

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
  pdfSentences: string[];
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
