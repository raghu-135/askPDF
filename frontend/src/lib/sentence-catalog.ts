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

const boxesNearlyEqual = (left: HighlightBBox, right: HighlightBBox): boolean => (
  left.page === right.page
  && Math.abs(left.x - right.x) < 1
  && Math.abs(left.y - right.y) < 1
  && Math.abs(left.width - right.width) < 1
  && Math.abs(left.height - right.height) < 1
);

const boxContains = (outer: HighlightBBox, inner: HighlightBBox, tolerance = 1): boolean => (
  outer.page === inner.page
  && inner.x >= outer.x - tolerance
  && inner.y >= outer.y - tolerance
  && inner.x + inner.width <= outer.x + outer.width + tolerance
  && inner.y + inner.height <= outer.y + outer.height + tolerance
);

export function uniqueHighlightBoxes<T extends HighlightBBox>(boxes: readonly T[]): T[] {
  const sorted = [...boxes].sort(
    (left, right) => (right.width * right.height) - (left.width * left.height),
  );
  const kept: T[] = [];
  for (const box of sorted) {
    const redundant = kept.some((existing) => (
      boxesNearlyEqual(existing, box) || boxContains(existing, box)
    ));
    if (!redundant) {
      kept.push(box);
    }
  }
  return kept;
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
