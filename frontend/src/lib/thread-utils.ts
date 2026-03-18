import { PdfTab } from "../components/PdfTabs";
import { Thread, addFileToThread, getThread, getPdfByHash, removeSourceFromThread } from "./api";

/**
 * Loads all PDFs and web sources for a thread and returns PdfTabs.
 */
export async function loadThreadTabs(thread: Thread, apiBase: string): Promise<PdfTab[]> {
  const threadData = await getThread(thread.id);
  if (!threadData.files || threadData.files.length === 0) return [];
  const loadedTabs: PdfTab[] = [];
  for (const file of threadData.files) {
    const isWeb = file.source_type === 'web';
    if (isWeb) {
      // Web source tab — no PDF to display, just record it
      loadedTabs.push({
        id: file.file_hash,
        fileName: file.file_name,
        fileHash: file.file_hash,
        pdfUrl: '',
        sentences: [],
        text: '',
        sourceType: 'web',
        sourceUrl: file.file_path || file.file_name,
      });
    } else {
      try {
        const pdfData = await getPdfByHash(file.file_hash);
        loadedTabs.push({
          id: file.file_hash,
          fileName: file.file_name,
          fileHash: file.file_hash,
          pdfUrl: `${apiBase}${pdfData.pdfUrl}?t=${Date.now()}`,
          sentences: pdfData.sentences,
          text: extractTextFromSentences(pdfData.sentences),
          sourceType: 'pdf',
        });
      } catch (err) {
        // Optionally log error
      }
    }
  }
  return loadedTabs;
}

/**
 * Creates a PdfTab from upload data.
 */
export function createPdfTabFromUpload(data: any, apiBase: string): PdfTab {
  return {
    id: data?.fileHash || `tab-${Date.now()}`,
    fileName: data?.fileName || 'Untitled.pdf',
    fileHash: data?.fileHash || '',
    pdfUrl: data?.pdfUrl ? `${apiBase}${data.pdfUrl}?t=${Date.now()}` : '',
    sentences: data?.sentences || [],
    text: extractTextFromSentences(data?.sentences || []),
    sourceType: 'pdf',
  };
}

/**
 * Creates a PdfTab from a web source indexing result.
 */
export function createWebTabFromIndexed(fileHash: string, url: string): PdfTab {
  return {
    id: fileHash,
    fileName: url,
    fileHash,
    pdfUrl: '',
    sentences: [],
    text: '',
    sourceType: 'web',
    sourceUrl: url,
  };
}

/**
 * Extracts text from an array of sentences.
 */
export function extractTextFromSentences(sentences: any[]): string {
  return (sentences || []).map((s: any) => s.text).join(' ');
}
/**
 * Orchestrates selecting a thread: clears current state and loads its context.
 */
export async function handleThreadSelectUtil(
  thread: Thread | null,
  apiBase: string,
  setPdfTabs: (tabs: PdfTab[]) => void,
  setActiveTabId: (id: string | null) => void,
  setCurrentPdfId: (id: number | null) => void,
  setCurrentChatId: (id: number | null) => void,
  setPlayRequestId: (id: number | null) => void,
  setActiveSource: (src: 'pdf' | 'chat') => void,
  setChatSentences: (sentences: any[]) => void,
  setHighlightedSourceSentenceIds: (ids: number[]) => void,
  setHighlightedSourceFileHash: (hash: string | null) => void,
  setHighlightedSourceMessageId: (id: string | null) => void,
  setActiveThread: (thread: Thread | null) => void,
  setIsPdfLoading: (loading: boolean) => void
) {
  // Clear current state
  setPdfTabs([]);
  setActiveTabId(null);
  setCurrentPdfId(null);
  setCurrentChatId(null);
  setPlayRequestId(null);
  setActiveSource('pdf');
  setChatSentences([]);
  setHighlightedSourceSentenceIds([]);
  setHighlightedSourceFileHash(null);
  setHighlightedSourceMessageId(null);

  if (thread) {
    try {
      setIsPdfLoading(true);
      const detailedThread = await getThread(thread.id);
      setActiveThread(detailedThread);

      const loadedTabs = await loadThreadTabs(detailedThread, apiBase);
      if (loadedTabs.length > 0) {
        setPdfTabs(loadedTabs);
        setActiveTabId(loadedTabs[0].id);
      }
    } catch (err) {
      console.error('Failed to load thread files:', err);
    } finally {
      setIsPdfLoading(false);
    }
  } else {
    setActiveThread(null);
  }
}

/**
 * Handles new PDF uploads by creating a tab and updating the thread.
 */
export async function handlePdfUploadedUtil(
  data: any,
  apiBase: string,
  activeThread: Thread | null,
  setPdfTabs: (update: (prev: PdfTab[]) => PdfTab[]) => void,
  setActiveTabId: (id: string | null) => void,
  setActiveThread: (thread: Thread | null) => void,
  setSidebarVersion: (update: (v: number) => number) => void,
  setCurrentPdfId: (id: number | null) => void,
  setCurrentChatId: (id: number | null) => void,
  setPlayRequestId: (id: number | null) => void,
  setActiveSource: (src: 'pdf' | 'chat') => void
) {
  const newTab = createPdfTabFromUpload(data, apiBase);

  setPdfTabs(prev => [...prev, newTab]);
  setActiveTabId(newTab.id);

  if (activeThread && data?.fileHash && data?.fileName) {
    try {
      await addFileToThread(activeThread.id, data.fileHash, data.fileName, newTab.text);
      const updatedThread = await getThread(activeThread.id);
      setActiveThread(updatedThread);
      setSidebarVersion(v => v + 1);
    } catch (error) {
      console.error('Failed to add file to thread:', error);
    }
  }

  setCurrentPdfId(null);
  setCurrentChatId(null);
  setPlayRequestId(null);
  setActiveSource('pdf');
}

/**
 * Handles new web sources by creating a tab and updating the thread.
 */
export async function handleWebIndexedUtil(
  data: { fileHash: string; url: string; status: string; message?: string },
  activeThread: Thread | null,
  setPdfTabs: (update: (prev: PdfTab[]) => PdfTab[]) => void,
  setActiveTabId: (id: string | null) => void,
  setActiveThread: (thread: Thread | null) => void,
  setSidebarVersion: (update: (v: number) => number) => void,
  setCurrentPdfId: (id: number | null) => void,
  setCurrentChatId: (id: number | null) => void,
  setPlayRequestId: (id: number | null) => void,
  setActiveSource: (src: 'pdf' | 'chat') => void
) {
  if (data.status !== 'accepted' || !activeThread || !data.fileHash) return;

  const newTab = createWebTabFromIndexed(data.fileHash, data.url);

  setPdfTabs(prev => {
    if (prev.some(t => t.id === newTab.id)) return prev;
    return [...prev, newTab];
  });
  setActiveTabId(newTab.id);

  try {
    const updatedThread = await getThread(activeThread.id);
    setActiveThread(updatedThread);
    setSidebarVersion(v => v + 1);
  } catch (error) {
    console.error('Failed to refresh thread after web source:', error);
  }

  setCurrentPdfId(null);
  setCurrentChatId(null);
  setPlayRequestId(null);
  setActiveSource('pdf');
}

/**
 * Orchestrates total removal of a source from a thread.
 */
export async function handleTabRemoveUtil(
  tabId: string,
  pdfTabs: PdfTab[],
  activeThread: Thread | null,
  handleTabClose: (id: string) => void,
  setActiveThread: (thread: Thread | null) => void,
  setSidebarVersion: (update: (v: number) => number) => void
) {
  if (!activeThread) return;
  const tab = pdfTabs.find(t => t.id === tabId);
  if (!tab) return;

  try {
    await removeSourceFromThread(activeThread.id, tab.fileHash);
    handleTabClose(tabId);
    const updatedThread = await getThread(activeThread.id);
    setActiveThread(updatedThread);
    setSidebarVersion(v => v + 1);
  } catch (error) {
    console.error('Failed to manage source removal:', error);
  }
}
