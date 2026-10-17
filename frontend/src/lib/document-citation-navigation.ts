import type { DocumentCanvasCitationTarget } from './canvas-spec';

const toSentenceId = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isInteger(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    if (Number.isInteger(parsed)) return parsed;
  }
  return null;
};

export const citationHighlightIds = (target: DocumentCanvasCitationTarget): number[] => {
  if (target.sentenceIds?.length) {
    const ids = target.sentenceIds
      .map((value) => toSentenceId(value))
      .filter((value): value is number => value != null);
    if (ids.length) return ids;
  }
  const sentenceId = toSentenceId(target.sentenceId);
  if (sentenceId != null) {
    return [sentenceId];
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
