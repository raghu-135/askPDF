import type { DocumentCanvasCitationTarget } from './canvas-spec';

export const citationHighlightIds = (target: DocumentCanvasCitationTarget): number[] => {
  if (target.sentenceIds?.length) {
    return target.sentenceIds.filter((value) => Number.isInteger(value));
  }
  if (target.sentenceId != null && Number.isInteger(target.sentenceId)) {
    return [target.sentenceId];
  }
  return [];
};

export const shouldPreserveCitationHighlight = (
  pending: DocumentCanvasCitationTarget | null,
  documentTab: { fileHash: string } | null | undefined,
): boolean => Boolean(
  pending
  && documentTab
  && pending.fileHash === documentTab.fileHash,
);
