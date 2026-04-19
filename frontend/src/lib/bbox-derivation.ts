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

export function deriveBBoxes(sentence: BackendSentence): BBox[] {
  // Use bboxes array if available (multi-line sentences), otherwise use single bbox
  const bboxSources = sentence.bboxes || [sentence.bbox];

  return bboxSources.map((bbox) => {
    // Handle both old format [x0, y0, x1, y1] and new format with page info
    if (Array.isArray(bbox)) {
      // Old format: [x0, y0, x1, y1]
      const [x0, y0, x1, y1] = bbox;
      return {
        page: sentence.page,
        x: x0,
        y: y0,  // Already top-left origin from backend
        width: x1 - x0,
        height: y1 - y0,
        page_width: sentence.page_width,
        page_height: sentence.page_height,
      };
    } else {
      // New format: { page, x0, y0, x1, y1, page_width, page_height }
      return {
        page: bbox.page,
        x: bbox.x0,
        y: bbox.y0,
        width: bbox.x1 - bbox.x0,
        height: bbox.y1 - bbox.y0,
        page_width: bbox.page_width,
        page_height: bbox.page_height,
      };
    }
  });
}

/**
 * Transform backend sentences to frontend format by deriving bboxes from bbox
 */
export function transformSentences(backendSentences: BackendSentence[]): Array<Omit<BackendSentence, 'bboxes'> & { bboxes: BBox[] }> {
  console.log('[bbox-derivation] Transforming sentences:', backendSentences.length);
  const transformed = backendSentences.map(sentence => {
    const { bboxes: _originalBboxes, ...rest } = sentence;
    return {
      ...rest,
      bboxes: deriveBBoxes(sentence),
    };
  });
  console.log('[bbox-derivation] Sample transformed sentence:', transformed[0]);
  return transformed;
}
