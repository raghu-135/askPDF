export type CatalogSentence = {
  id: number;
  readable?: boolean;
};

export function sentenceById<T extends CatalogSentence>(
  sentences: T[] | null,
  id: number | null,
): T | undefined {
  if (id === null || !sentences) return undefined;
  return sentences.find((sentence) => sentence.id === id);
}

export function highlightIdSet(
  currentId: number | null,
  highlightSentenceIds?: number[] | null,
): Set<number> {
  const ids = highlightSentenceIds?.length
    ? highlightSentenceIds
    : (currentId !== null ? [currentId] : []);
  return new Set(ids);
}

export function readableSentenceIds(sentences: CatalogSentence[] | null): number[] {
  if (!sentences) return [];
  return sentences.filter((sentence) => sentence.readable !== false).map((sentence) => sentence.id);
}

export function adjacentReadableId(
  sentences: CatalogSentence[] | null,
  currentId: number,
  direction: -1 | 1,
): number | null {
  const playlist = readableSentenceIds(sentences);
  const index = playlist.indexOf(currentId);
  if (index < 0) return null;
  const nextIndex = index + direction;
  if (nextIndex < 0 || nextIndex >= playlist.length) return null;
  return playlist[nextIndex];
}
