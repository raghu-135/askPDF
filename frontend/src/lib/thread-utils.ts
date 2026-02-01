import { PdfTab } from "../components/PdfTabs";
import { Thread, addFileToThread, getThread, getPdfByHash } from "./api";

/**
 * Loads all PDFs for a thread and returns PdfTabs.
 */
export async function loadThreadTabs(thread: Thread, apiBase: string): Promise<PdfTab[]> {
  const threadData = await getThread(thread.id);
  if (!threadData.files || threadData.files.length === 0) return [];
  const loadedTabs: PdfTab[] = [];
  for (const file of threadData.files) {
    try {
      const pdfData = await getPdfByHash(file.file_hash);
      loadedTabs.push({
        id: file.file_hash,
        fileName: file.file_name,
        fileHash: file.file_hash,
        pdfUrl: `${apiBase}${pdfData.pdfUrl}?t=${Date.now()}`,
        sentences: pdfData.sentences,
        text: extractTextFromSentences(pdfData.sentences),
      });
    } catch (err) {
      // Optionally log error
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
  };
}

/**
 * Extracts text from an array of sentences.
 */
export function extractTextFromSentences(sentences: any[]): string {
  return (sentences || []).map((s: any) => s.text).join(' ');
}
