import { clampDecisionPanelRatio } from './conversation-ui-state.ts';

/** Minimum scroll height for the limitations block inside the decision shell. */
export const REVIEW_DECISION_PANEL_LIMITATIONS_MIN_PX = 120;

/** Column-fill panel: limitations row keeps the flexible slice before the provisional body. */
export const REVIEW_DECISION_PANEL_SCROLL_GRID_ROWS =
  `auto auto minmax(${REVIEW_DECISION_PANEL_LIMITATIONS_MIN_PX}px, 1fr) minmax(0, auto)`;

/** Minimum height for the scroll grid (handle + title + limitations floor). */
export const REVIEW_DECISION_PANEL_SCROLL_MIN_PX =
  REVIEW_DECISION_PANEL_LIMITATIONS_MIN_PX + 56;

/** Footer never grows past this; guidance scrolls inside, buttons stay in the band. */
export const REVIEW_DECISION_PANEL_FOOTER_MAX_HEIGHT = 'min(30vh, 136px)';

/** While a review decision is open, cap transcript height so the panel keeps the column. */
export const REVIEW_PANEL_OPEN_TRANSCRIPT_MAX_PX = 72;

/** Expanded transcript scroll cap so the decision column keeps the footer. */
export const REVIEW_TRANSCRIPT_EXPANDED_MAX_HEIGHT = 'min(28vh, 220px)';

/** Stacked workbench (chat under thread pane): keep transcript tiny so the panel fits. */
export const REVIEW_STACKED_TRANSCRIPT_MAX_HEIGHT = `${REVIEW_PANEL_OPEN_TRANSCRIPT_MAX_PX}px`;

/** Floor for the decision column while review is open (footer + limitations). */
export const REVIEW_DECISION_COLUMN_MIN_HEIGHT = 'min(36vh, 240px)';

/** Stacked workbench needs a taller decision slice at 1024×768 and below. */
export const REVIEW_STACKED_DECISION_COLUMN_MIN_HEIGHT = 'min(46vh, 300px)';

/** Non-fill panels: handle, title, header, body, footer in one grid. */
export const REVIEW_DECISION_PANEL_GRID_ROWS =
  `${REVIEW_DECISION_PANEL_SCROLL_GRID_ROWS} auto`;

export const REVIEW_LIMITATIONS_MAX_HEIGHT = 'min(24vh, 180px)';

/** Keeps review action buttons in view while limitations and budget copy scroll above. */
export const REVIEW_DECISION_PANEL_FIXED_HEADER_MAX_HEIGHT = 'min(36vh, 260px)';

export const REVIEW_PANEL_MIN_TRANSCRIPT_RATIO = 0.12;

/** Minimum stacked workbench chat slice while reviewing (under documents). */
export const REVIEW_STACKED_WORKBENCH_BOTTOM_RATIO = 0.58;

export const DEEP_RESEARCH_REVIEW_WORKBENCH_BOOST_EVENT = 'askpdf-workbench-review-boost';

/** Match workbench auto placement: below this width the chat column stacks under documents. */
export { WORKBENCH_AUTO_SIDE_MIN_WIDTH } from './workbench-layout.ts';

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
