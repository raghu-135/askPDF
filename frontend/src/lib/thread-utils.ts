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
  for (const file of threadData.files) {
    try {
      const pdfData = await getPdfByHash(file.file_hash, thread.id);
      const transformedSentences = transformSentences(pdfData.sentences);
      loadedTabs.push({
        id: file.file_hash,
        fileName: file.file_name,
        fileHash: file.file_hash,
        pdfUrl: `${API_BASE}/api${pdfData.pdfUrl}?t=${Date.now()}`,
        sentences: transformedSentences,
        text: extractTextFromSentences(transformedSentences),
        sourceType: 'pdf',
      });
    } catch (err) {
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
    pdfUrl: data?.pdfUrl ? `${API_BASE}/api${data.pdfUrl}?t=${Date.now()}` : '',
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
