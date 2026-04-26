import { PdfTab } from "../components/PdfTabs";
import { Thread, getThread, getPdfByHash, API_BASE } from "./api";
import { transformSentences } from "./bbox-derivation";

/**
 * Loads all sources (PDFs and web-converted PDFs) for a thread and returns PdfTabs.
 *
 * With unified PDF flow, all sources are treated identically - web pages are
 * converted to PDFs on the backend and served through the same endpoints.
 */
export async function loadThreadTabs(thread: Thread): Promise<PdfTab[]> {
  const threadData = await getThread(thread.id);
  if (!threadData.files || threadData.files.length === 0) return [];
  const loadedTabs: PdfTab[] = [];
  for (const file of threadData.files) {
    // With unified flow, all sources are PDFs (both uploaded and web-converted)
    try {
      const pdfData = await getPdfByHash(file.file_hash, thread.id);
      const transformedSentences = transformSentences(pdfData.sentences);
      loadedTabs.push({
        id: file.file_hash,
        fileName: file.file_name,
        fileHash: file.file_hash,
        pdfUrl: `${API_BASE}${pdfData.pdfUrl}?t=${Date.now()}`,
        sentences: transformedSentences,
        text: extractTextFromSentences(transformedSentences),
        sourceType: file.source_type === 'web' ? 'web' : 'pdf',
        sourceUrl: file.source_type === 'web' ? file.file_path || file.file_name : undefined,
      });
    } catch (err) {
      // Optionally log error - skip files that can't be loaded
      console.warn(`Failed to load file ${file.file_hash}:`, err);
    }
  }
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
    pdfUrl: data?.pdfUrl ? `${API_BASE}${data.pdfUrl}?t=${Date.now()}` : '',
    sentences: sentences ? transformedSentences : null,
    text: sentences ? extractTextFromSentences(transformedSentences) : '',
    sourceType: 'pdf',
    parsingStatus: sentences ? 'completed' : 'pending',
  };
}

/**
 * Creates a PdfTab from a web source indexing result.
 *
 * With unified PDF flow, web sources are converted to PDFs on the backend
 * and treated identically to uploaded PDFs.
 */
export function createWebTabFromIndexed(fileHash: string, url: string, threadId: string, title?: string): PdfTab {
  return {
    id: fileHash,
    fileName: title || url,
    fileHash,
    pdfUrl: `${API_BASE}/api/threads/${threadId}/files/${fileHash}/download?t=${Date.now()}`,
    sentences: [],  // Will be populated by getPdfByHash on thread load
    text: '',
    sourceType: 'web',
    sourceUrl: url,  // Original URL for reference
  };
}

/**
 * Extracts text from an array of sentences.
 */
export function extractTextFromSentences(sentences: any[]): string {
  return (sentences || []).map((s: any) => s.text).join(' ');
}
