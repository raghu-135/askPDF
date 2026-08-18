export type PanelRatioConfig = {
  default: number;
  min: number;
  max: number;
};

export const STANDARD_PANEL_RATIO = {
  default: 0.32,
  min: 0.2,
  max: 0.8,
} satisfies PanelRatioConfig;

export const PANEL_RATIOS = {
  chat: STANDARD_PANEL_RATIO,
  pdfSidebar: STANDARD_PANEL_RATIO,
  graphElements: STANDARD_PANEL_RATIO,
} satisfies Record<string, PanelRatioConfig>;

export const clampPanelRatio = (ratio: number, config: PanelRatioConfig) => (
  Math.max(config.min, Math.min(config.max, ratio))
);

export const getHorizontalDragRatio = (
  currentClientX: number,
  startClientX: number,
  containerWidth: number
) => {
  if (containerWidth <= 0) return 0;
  return (currentClientX - startClientX) / containerWidth;
};
