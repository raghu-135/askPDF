export type CatalogSentence = {
  id: number;
  readable?: boolean;
};

export type HighlightBBox = {
  page: number;
  x: number;
  y: number;
  width: number;
  height: number;
};

export type SentenceWithBBoxes = CatalogSentence & {
  bboxes?: HighlightBBox[];
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
  if (!playlist.length) return null;
  const index = playlist.indexOf(currentId);
  if (index < 0) {
    if (direction > 0) {
      return playlist.find((id) => id > currentId) ?? null;
    }
    const reversed = [...playlist].reverse();
    return reversed.find((id) => id < currentId) ?? null;
  }
  const nextIndex = index + direction;
  if (nextIndex < 0 || nextIndex >= playlist.length) return null;
  return playlist[nextIndex];
}

export function firstHighlightableBBox(
  sentences: SentenceWithBBoxes[] | null,
  sentenceIds: number[],
): HighlightBBox | null {
  if (!sentences?.length || !sentenceIds.length) return null;
  for (const id of sentenceIds) {
    const sentence = sentenceById(sentences, id);
    for (const bbox of sentence?.bboxes ?? []) {
      if (bbox.width > 0 && bbox.height > 0) {
        return bbox;
      }
    }
  }
  return null;
}
