import { clampDecisionPanelRatio } from './conversation-ui-state.ts';

/** Grid contract: handle, title, bounded header, scroll body, pinned footer. */
export const REVIEW_DECISION_PANEL_GRID_ROWS = 'auto auto minmax(0, auto) minmax(0, 1fr) auto';

export const REVIEW_LIMITATIONS_MAX_HEIGHT = 'min(24vh, 180px)';

/** Keeps review action buttons in view while limitations and budget copy scroll above. */
export const REVIEW_DECISION_PANEL_FIXED_HEADER_MAX_HEIGHT = 'min(36vh, 260px)';

export const REVIEW_PANEL_MIN_TRANSCRIPT_RATIO = 0.12;

/** Sticky review footer (guidance field + two actions + helper copy). */
export const REVIEW_PANEL_DECISION_FOOTER_RESERVED_PX = 122;

/** Resize handle plus centered title row inside the decision shell. */
export const REVIEW_PANEL_DECISION_CHROME_PX = 40;

/**
 * Header, status chrome (lock alert, budget chips, canvases), and composer
 * outside the transcript/decision flex split (measured on the review layout fixture).
 */
export const REVIEW_PANEL_WORKBENCH_CHROME_PX = 308;

export const REVIEW_PANEL_TRANSCRIPT_MIN_PX = 72;

export function reviewPanelFlexWeights(panelRatio: number): { transcript: number; decision: number } {
  const decision = clampDecisionPanelRatio(panelRatio);
  const transcript = Math.max(REVIEW_PANEL_MIN_TRANSCRIPT_RATIO, 1 - decision);
  const total = transcript + decision;
  return { transcript: transcript / total, decision: decision / total };
}

export type ReviewPanelViewportChrome = {
  workbenchChromePx: number;
  transcriptMinPx: number;
  decisionChromePx: number;
  footerReservedPx: number;
};

export const DEFAULT_REVIEW_PANEL_VIEWPORT_CHROME: ReviewPanelViewportChrome = {
  workbenchChromePx: REVIEW_PANEL_WORKBENCH_CHROME_PX,
  transcriptMinPx: REVIEW_PANEL_TRANSCRIPT_MIN_PX,
  decisionChromePx: REVIEW_PANEL_DECISION_CHROME_PX,
  footerReservedPx: REVIEW_PANEL_DECISION_FOOTER_RESERVED_PX,
};

/** Layout contract: pinned footer stays inside the decision flex slice at common viewports. */
export function reviewPanelLayoutFitsViewport(
  weights: { transcript: number; decision: number },
  viewportHeightPx: number,
  chrome: Partial<ReviewPanelViewportChrome> = {},
): boolean {
  const resolved: ReviewPanelViewportChrome = { ...DEFAULT_REVIEW_PANEL_VIEWPORT_CHROME, ...chrome };
  const flexRegionPx = Math.max(0, viewportHeightPx - resolved.workbenchChromePx);
  const transcriptSharePx = weights.transcript * flexRegionPx;
  const transcriptPx = Math.max(resolved.transcriptMinPx, transcriptSharePx);
  const decisionShellPx = Math.max(0, flexRegionPx - transcriptPx);
  return decisionShellPx >= resolved.decisionChromePx + resolved.footerReservedPx;
}
