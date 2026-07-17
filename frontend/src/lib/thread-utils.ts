import { PdfTab } from "../components/PdfTabs";
import { Thread, getThread, getPdfByHash, API_BASE } from "./api";
import { transformSentences } from "./bbox-derivation";

/**
 * Loads all PDF sources for a thread and returns PdfTabs.
 */
export async function loadThreadTabs(thread: Thread): Promise<PdfTab[]> {
  const threadData = await getThread(thread.id);
  if (!threadData.files || threadData.files.length === 0) return [];
  const loadedTabs: PdfTab[] = [];
  
  // Process files in the order returned by backend (already ordered by added_at DESC)
  for (const threadFile of threadData.files) {
    try {
      const pdfData = await getPdfByHash(threadFile.fileHash, thread.id);
      const transformedSentences = transformSentences(pdfData.sentences);
      loadedTabs.push({
        id: threadFile.fileHash,
        fileName: threadFile.fileName,
        fileHash: threadFile.fileHash,
        downloadUrl: `${API_BASE}/api${pdfData.downloadUrl}?t=${Date.now()}`,
        sentences: transformedSentences,
        text: extractTextFromSentences(transformedSentences),
        sourceType: 'pdf',
        parsingStatus: 'completed',
      });
    } catch (err) {
      console.warn(`Failed to load file ${threadFile.fileHash}, creating tab with pending status:`, err);
      // Create tab with basic info even if API call fails
      loadedTabs.push({
        id: threadFile.fileHash,
        fileName: threadFile.fileName,
        fileHash: threadFile.fileHash,
        downloadUrl: `${API_BASE}/api/threads/${thread.id}/files/${threadFile.fileHash}/download?t=${Date.now()}`,
        sentences: null,
        text: '',
        sourceType: 'pdf',
        parsingStatus: 'pending',
      });
    }
  }
  
  // Return tabs in the same order as backend (most recent first)
  return loadedTabs;
}

/**
 * Creates a PdfTab from upload data.
 */
export function createPdfTabFromUpload(data: any): PdfTab {
  const sentences = data?.sentences;
  const transformedSentences = sentences ? transformSentences(sentences) : [];
  return {
    id: data?.fileHash || `tab-${Date.now()}`,
    fileName: data?.fileName || 'Untitled.pdf',
    fileHash: data?.fileHash || '',
    downloadUrl: data?.downloadUrl ? `${API_BASE}/api${data.downloadUrl}?t=${Date.now()}` : '',
    sentences: sentences ? transformedSentences : null,
    text: sentences ? extractTextFromSentences(transformedSentences) : '',
    sourceType: 'pdf',
    parsingStatus: sentences ? 'completed' : 'pending',
  };
}

/**
 * Extracts text from an array of sentences.
 */
export function extractTextFromSentences(sentences: any[]): string {
  return (sentences || []).map((s: any) => s.text).join(' ');
}
