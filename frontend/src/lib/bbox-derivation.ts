/**
 * Derive bboxes array from backend sentence format
 * Backend outputs: { id, text, label, page, bbox: [x0, y0, x1, y1] or object, page_width, page_height, words, font }
 * Frontend needs: bboxes: [{ x, y, width, height, page_width, page_height, page }]
 */

export type BBox = {
  page: number;
  x: number;
  y: number;
  width: number;
  height: number;
  page_height: number;
  page_width: number;
};

export type Word = {
  text: string;
  x0: number;
  x1: number;
  top: number;
  bottom: number;
  page_width: number;
  page_height: number;
  char_start: number;
  char_end: number;
  page?: number;  // Added for multi-page sentences
};

export type BackendSentence = {
  id: number;
  text: string;
  label: string;
  page: number;
  bbox: [number, number, number, number] | { page: number; x0: number; y0: number; x1: number; y1: number; page_width: number; page_height: number };
  bboxes?: ([number, number, number, number] | { page: number; x0: number; y0: number; x1: number; y1: number; page_width: number; page_height: number })[];  // Multiple line bboxes with page info
  page_width: number;
  page_height: number;
  words?: Word[];
  font?: { name: string; size: number };
  pages?: number[];  // For multi-page sentences
};

const positivePageDimension = (value: unknown): number | null => {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
};

export function deriveBBoxes(sentence: BackendSentence): BBox[] {
  // Use bboxes array if available (multi-line sentences), otherwise use single bbox
  const bboxSources = sentence.bboxes || [sentence.bbox];

  // Filter out null/undefined bboxes (web content may not have bounding boxes)
  const validBboxes = bboxSources.filter((bbox): bbox is NonNullable<typeof bbox> => bbox != null);

  if (validBboxes.length === 0) {
    return [];
  }

  const derived: BBox[] = [];
  for (const bbox of validBboxes) {
    // Handle both old format [x0, y0, x1, y1] and new format with page info
    if (Array.isArray(bbox)) {
      const [x0, y0, x1, y1] = bbox;
      const pageWidth = positivePageDimension(sentence.page_width);
      const pageHeight = positivePageDimension(sentence.page_height);
      if (!pageWidth || !pageHeight) {
        continue;
      }
      derived.push({
        page: sentence.page,
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
        page_width: pageWidth,
        page_height: pageHeight,
      });
      continue;
    }
    const pageWidth = positivePageDimension(bbox.page_width ?? sentence.page_width);
    const pageHeight = positivePageDimension(bbox.page_height ?? sentence.page_height);
    if (!pageWidth || !pageHeight) {
      continue;
    }
    derived.push({
      page: bbox.page,
      x: bbox.x0,
      y: bbox.y0,
      width: bbox.x1 - bbox.x0,
      height: bbox.y1 - bbox.y0,
      page_width: pageWidth,
      page_height: pageHeight,
    });
  }
  return derived;
}

/**
 * Transform backend sentences to frontend format by deriving bboxes from bbox
 */
export function isParsedSentencePayload(sentences: unknown): sentences is BackendSentence[] {
  return Array.isArray(sentences) && sentences.length > 0;
}

export function transformSentences(backendSentences: BackendSentence[] | null | undefined): Array<Omit<BackendSentence, 'bboxes'> & { bboxes: BBox[] }> {
  if (!Array.isArray(backendSentences)) {
    return [];
  }
  return backendSentences.map(sentence => {
    const { bboxes: _originalBboxes, ...rest } = sentence;
    return {
      ...rest,
      bboxes: deriveBBoxes(sentence),
    };
  });
}
