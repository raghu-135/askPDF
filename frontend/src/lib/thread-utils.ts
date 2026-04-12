import { PdfTab } from "../components/PdfTabs";
import { Thread, addFileToThread, getThread, getPdfByHash } from "./api";

/**
 * Loads all sources (PDFs and web-converted PDFs) for a thread and returns PdfTabs.
 *
 * With unified PDF flow, all sources are treated identically - web pages are
 * converted to PDFs on the backend and served through the same endpoints.
 */
export async function loadThreadTabs(thread: Thread, apiBase: string): Promise<PdfTab[]> {
  const threadData = await getThread(thread.id);
  if (!threadData.files || threadData.files.length === 0) return [];
  const loadedTabs: PdfTab[] = [];
  for (const file of threadData.files) {
    // With unified flow, all sources are PDFs (both uploaded and web-converted)
    try {
      const pdfData = await getPdfByHash(file.file_hash);
      loadedTabs.push({
        id: file.file_hash,
        fileName: file.file_name,
        fileHash: file.file_hash,
        pdfUrl: `${apiBase}${pdfData.pdfUrl}?t=${Date.now()}`,
        sentences: pdfData.sentences,
        text: extractTextFromSentences(pdfData.sentences),
        sourceType: 'pdf',  // Unified type
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
 *
 * With unified PDF flow, web sources are converted to PDFs on the backend
 * and treated identically to uploaded PDFs.
 */
export function createWebTabFromIndexed(fileHash: string, url: string, apiBase: string, title?: string): PdfTab {
  return {
    id: fileHash,
    fileName: title || url,
    fileHash,
    pdfUrl: `${apiBase}/api/pdf-file/${fileHash}?t=${Date.now()}`,
    sentences: [],  // Will be populated by getPdfByHash on thread load
    text: '',
    sourceType: 'pdf',  // Unified as PDF
    sourceUrl: url,  // Original URL for reference
  };
}

/**
 * Extracts text from an array of sentences.
 */
export function extractTextFromSentences(sentences: any[]): string {
  return (sentences || []).map((s: any) => s.text).join(' ');
}
