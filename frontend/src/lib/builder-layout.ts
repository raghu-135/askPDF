import { clampPanelRatio, PANEL_RATIOS } from './panel-ratio.ts';

export const BUILDER_LAYOUT_STORAGE_KEY = 'askpdf.agentWorkflowBuilder.layout.v1';

export const DEFAULT_BUILDER_LAYOUT = {
  graphElementsRatio: PANEL_RATIOS.graphElements.default,
  graphElementsCollapsed: false,
};

export const normalizeBuilderLayout = (value?: unknown) => {
  const stored = value && typeof value === 'object' ? value as Partial<typeof DEFAULT_BUILDER_LAYOUT> : {};
  return {
    graphElementsRatio: clampPanelRatio(Number(stored.graphElementsRatio) || PANEL_RATIOS.graphElements.default, PANEL_RATIOS.graphElements),
    graphElementsCollapsed: Boolean(stored.graphElementsCollapsed),
  };
};
