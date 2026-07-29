import type { PdfTab } from "./document-tabs.ts";
import type { Project, Thread, ThreadFile } from "./api.ts";
import { getThread, getPdfByHash, getPdfForTarget, getProjectFiles, API_BASE } from "./api.ts";
import { transformSentences } from "./bbox-derivation.ts";
import { ProcessStatus, ThreadFileSourceType } from "./enums.ts";

export type DetailedThread = Thread & { files?: ThreadFile[] };

const createPendingThreadPdfTab = (threadId: string, threadFile: ThreadFile): PdfTab => ({
  id: threadFile.fileHash,
  fileName: threadFile.fileName,
  fileHash: threadFile.fileHash,
  downloadUrl: `${API_BASE}/api/threads/${threadId}/files/${threadFile.fileHash}/download?t=${Date.now()}`,
  sentences: null,
  text: '',
  sourceType: threadFile.sourceType || ThreadFileSourceType.Pdf,
  parsingStatus: threadFile.processingStatus || ProcessStatus.Pending,
  processingError: threadFile.processingError,
  associationScope: threadFile.associationScope,
  isProjectKnowledge: threadFile.isProjectKnowledge,
});

export async function hydrateThreadPdfTab(threadId: string, threadFile: ThreadFile): Promise<PdfTab> {
  const pdfData = await getPdfByHash(threadFile.fileHash, threadId);
  const transformedSentences = transformSentences(pdfData.sentences);
  return {
    id: threadFile.fileHash,
    fileName: threadFile.fileName,
    fileHash: threadFile.fileHash,
    downloadUrl: `${API_BASE}/api${pdfData.downloadUrl}?t=${Date.now()}`,
    sentences: transformedSentences,
    text: extractTextFromSentences(transformedSentences),
    sourceType: threadFile.sourceType || ThreadFileSourceType.Pdf,
    parsingStatus: threadFile.processingStatus || ProcessStatus.Completed,
    processingError: threadFile.processingError,
    associationScope: threadFile.associationScope,
    isProjectKnowledge: threadFile.isProjectKnowledge,
  };
}

export async function loadProjectTabs(project: Project, options?: { eagerCount?: number }): Promise<PdfTab[]> {
  const { files } = await getProjectFiles(project.id);
  const eagerCount = Math.max(0, options?.eagerCount ?? 1);
  return Promise.all(files.map(async (file, index) => {
    const pending: PdfTab = {
      id: file.fileHash,
      fileName: file.fileName,
      fileHash: file.fileHash,
      downloadUrl: `${API_BASE}/api/projects/${project.id}/files/${file.fileHash}/download?t=${Date.now()}`,
      sentences: null,
      text: '',
      sourceType: file.sourceType || ThreadFileSourceType.Pdf,
      parsingStatus: file.processingStatus || ProcessStatus.Pending,
      processingError: file.processingError,
      associationScope: 'project',
      isProjectKnowledge: true,
    };
    if (index >= eagerCount) return pending;
    try {
      const pdfData = await getPdfForTarget(file.fileHash, { scope: 'project', id: project.id });
      const sentences = transformSentences(pdfData.sentences);
      return {
        ...pending,
        downloadUrl: `${API_BASE}/api${pdfData.downloadUrl}?t=${Date.now()}`,
        sentences,
        text: extractTextFromSentences(sentences),
        parsingStatus: ProcessStatus.Completed,
      };
    } catch {
      return pending;
    }
  }));
}

/**
 * Loads PDF sources for a thread and returns PdfTabs. The first tab is hydrated
 * eagerly; later tabs are lightweight placeholders so selecting a thread is fast.
 */
export async function loadThreadTabs(thread: DetailedThread, options?: { eagerCount?: number }): Promise<PdfTab[]> {
  const threadData = Array.isArray(thread.files) ? thread : await getThread(thread.id);
  if (!threadData.files || threadData.files.length === 0) return [];
  const loadedTabs: PdfTab[] = [];
  const eagerCount = Math.max(0, options?.eagerCount ?? 1);
  
  // Process files in the order returned by backend (already ordered by added_at DESC)
  for (const [index, threadFile] of threadData.files.entries()) {
    if (index >= eagerCount) {
      loadedTabs.push(createPendingThreadPdfTab(threadData.id, threadFile));
      continue;
    }
    try {
      loadedTabs.push(await hydrateThreadPdfTab(threadData.id, threadFile));
    } catch (err) {
      console.warn(`Failed to load file ${threadFile.fileHash}, creating tab with pending status:`, err);
      loadedTabs.push(createPendingThreadPdfTab(threadData.id, threadFile));
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
    sourceType: ThreadFileSourceType.Pdf,
    parsingStatus: sentences ? ProcessStatus.Completed : ProcessStatus.Pending,
  };
}

/**
 * Extracts text from an array of sentences.
 */
export function extractTextFromSentences(sentences: any[]): string {
  return (sentences || []).map((s: any) => s.text).join(' ');
}
