import { clampDecisionPanelRatio } from './conversation-ui-state.ts';

/** Legacy grid contract (column-fill panels now use flex + bounded header). */
export const REVIEW_DECISION_PANEL_GRID_ROWS = 'auto auto auto minmax(0, 1fr) auto';

export const REVIEW_LIMITATIONS_MAX_HEIGHT = 'min(24vh, 180px)';

/** Keeps review action buttons in view while limitations and budget copy scroll above. */
export const REVIEW_DECISION_PANEL_FIXED_HEADER_MAX_HEIGHT = 'min(36vh, 260px)';

export const REVIEW_PANEL_MIN_TRANSCRIPT_RATIO = 0.12;

export function reviewPanelFlexWeights(panelRatio: number): { transcript: number; decision: number } {
  const decision = clampDecisionPanelRatio(panelRatio);
  const transcript = Math.max(REVIEW_PANEL_MIN_TRANSCRIPT_RATIO, 1 - decision);
  const total = transcript + decision;
  return { transcript: transcript / total, decision: decision / total };
}

/** Layout contract for 1024×768 review: footer slice must stay positive with many limitations. */
export function reviewPanelLayoutFitsViewport(
  weights: { transcript: number; decision: number },
  viewportHeightPx: number,
): boolean {
  const reservedFooterPx = 148;
  const reservedChromePx = 220;
  const decisionPx = weights.decision * viewportHeightPx;
  return decisionPx - reservedChromePx >= reservedFooterPx;
}
