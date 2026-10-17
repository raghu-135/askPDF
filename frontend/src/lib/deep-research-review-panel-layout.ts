import { clampDecisionPanelRatio } from './conversation-ui-state.ts';

/** CSS grid rows: resize handle, title, fixed header, scroll body, sticky footer. */
export const REVIEW_DECISION_PANEL_GRID_ROWS = 'auto auto auto minmax(0, 1fr) auto';

export const REVIEW_LIMITATIONS_MAX_HEIGHT = 'min(24vh, 180px)';

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
