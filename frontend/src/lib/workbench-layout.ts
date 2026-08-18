export type WorkbenchPlacement = 'auto' | 'left' | 'right' | 'bottom';
export type ResolvedWorkbenchPlacement = Exclude<WorkbenchPlacement, 'auto'>;

export type WorkbenchLayoutState = {
  placement: WorkbenchPlacement;
  visible: boolean;
  sideRatio: number;
  bottomRatio: number;
};

export const DEFAULT_WORKBENCH_LAYOUT: WorkbenchLayoutState = {
  placement: 'auto',
  visible: true,
  sideRatio: 0.32,
  bottomRatio: 0.4,
};

export const WORKBENCH_AUTO_SIDE_MIN_WIDTH = 1100;
export const WORKBENCH_HARD_SIDE_MIN_WIDTH = 720;

export const clampWorkbenchRatio = (
  ratio: number,
  minimum: number,
  maximum: number,
) => Math.max(minimum, Math.min(maximum, ratio));

export const normalizeWorkbenchLayout = (
  value?: Partial<WorkbenchLayoutState> & { dockMode?: string } | null,
): WorkbenchLayoutState => {
  const legacyDockMode = String(value?.dockMode || '');
  const rawPlacement = value?.placement || (
    ['auto', 'left', 'right', 'bottom'].includes(legacyDockMode)
      ? legacyDockMode
      : undefined
  );
  const placement = ['auto', 'left', 'right', 'bottom'].includes(String(rawPlacement))
    ? rawPlacement as WorkbenchPlacement
    : DEFAULT_WORKBENCH_LAYOUT.placement;
  const legacyHidden = legacyDockMode === 'hidden';

  return {
    placement,
    visible: legacyHidden ? false : value?.visible !== false,
    sideRatio: clampWorkbenchRatio(
      Number(value?.sideRatio) || DEFAULT_WORKBENCH_LAYOUT.sideRatio,
      0.2,
      0.8,
    ),
    bottomRatio: clampWorkbenchRatio(
      Number(value?.bottomRatio) || DEFAULT_WORKBENCH_LAYOUT.bottomRatio,
      0.25,
      0.6,
    ),
  };
};

export const readStoredWorkbenchLayout = (
  getItem: (key: string) => string | null,
  storageKey: string,
  fallback: WorkbenchLayoutState,
): WorkbenchLayoutState => {
  try {
    const stored = getItem(storageKey);
    return stored ? normalizeWorkbenchLayout(JSON.parse(stored)) : fallback;
  } catch {
    return fallback;
  }
};

export const resolveWorkbenchPlacement = (
  placement: WorkbenchPlacement,
  containerWidth: number,
  autoSideMinWidth = WORKBENCH_AUTO_SIDE_MIN_WIDTH,
  hardSideMinWidth = WORKBENCH_HARD_SIDE_MIN_WIDTH,
): ResolvedWorkbenchPlacement => {
  if (containerWidth > 0 && containerWidth < hardSideMinWidth) return 'bottom';
  if (placement === 'auto') {
    return containerWidth >= autoSideMinWidth ? 'right' : 'bottom';
  }
  return placement;
};

export const resizeWorkbenchRatio = ({
  placement,
  clientX,
  clientY,
  left,
  right,
  top,
  bottom,
}: {
  placement: ResolvedWorkbenchPlacement;
  clientX: number;
  clientY: number;
  left: number;
  right: number;
  top: number;
  bottom: number;
}) => {
  if (placement === 'bottom') {
    return clampWorkbenchRatio((bottom - clientY) / Math.max(1, bottom - top), 0.25, 0.6);
  }
  const raw = placement === 'left'
    ? (clientX - left) / Math.max(1, right - left)
    : (right - clientX) / Math.max(1, right - left);
  return clampWorkbenchRatio(raw, 0.2, 0.8);
};
