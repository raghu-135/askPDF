import { clampDecisionPanelRatio } from './conversation-ui-state.ts';

/** Grid contract: handle, title, bounded header, scroll body, pinned footer. */
export const REVIEW_DECISION_PANEL_GRID_ROWS = 'auto auto minmax(0, auto) minmax(0, 1fr) auto';

export const REVIEW_LIMITATIONS_MAX_HEIGHT = 'min(24vh, 180px)';

/** Keeps review action buttons in view while limitations and budget copy scroll above. */
export const REVIEW_DECISION_PANEL_FIXED_HEADER_MAX_HEIGHT = 'min(36vh, 260px)';

export const REVIEW_PANEL_MIN_TRANSCRIPT_RATIO = 0.12;

/** While a review decision is open, cap transcript height so the panel keeps the column. */
export const REVIEW_PANEL_OPEN_TRANSCRIPT_MAX_PX = 72;

export function reviewPanelFlexWeights(panelRatio: number): { transcript: number; decision: number } {
  const decision = clampDecisionPanelRatio(panelRatio);
  const transcript = Math.max(REVIEW_PANEL_MIN_TRANSCRIPT_RATIO, 1 - decision);
  const total = transcript + decision;
  return { transcript: transcript / total, decision: decision / total };
}

/**
 * Height left for the decision shell once review mode caps the transcript and
 * hides the lifecycle composer. Uses proportional chrome, not a fixed footer budget.
 */
export function reviewPanelOpenDecisionSlicePx(
  viewportHeightPx: number,
  chromeAboveDecisionPx: number,
): number {
  const transcriptPx = Math.min(REVIEW_PANEL_OPEN_TRANSCRIPT_MAX_PX, viewportHeightPx);
  return Math.max(0, viewportHeightPx - chromeAboveDecisionPx - transcriptPx);
}
