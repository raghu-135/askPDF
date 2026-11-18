import { useEffect, useState, type RefObject } from 'react';
import { clampDecisionPanelRatio } from '../lib/conversation-ui-state';

export function useRootPanelHeight(
  rootRef: RefObject<HTMLElement | null>,
  ratio: number,
  bounds: { min?: number; max?: number } = {},
) {
  const { min = 0.16, max = 0.82 } = bounds;
  const [rootHeight, setRootHeight] = useState(0);

  useEffect(() => {
    const element = rootRef.current;
    if (!element) return undefined;
    const sync = () => setRootHeight(element.getBoundingClientRect().height);
    sync();
    const observer = new ResizeObserver(sync);
    observer.observe(element);
    return () => observer.disconnect();
  }, [rootRef]);

  const clampedRatio = clampDecisionPanelRatio(ratio, min, max);
  const effectiveRoot = rootHeight > 0 ? rootHeight : 480;
  const panelHeight = Math.round(effectiveRoot * clampedRatio);
  const minHeightPx = Math.round(effectiveRoot * min);
  const maxHeightPx = Math.round(effectiveRoot * max);

  return { panelHeight, minHeightPx, maxHeightPx, rootHeight, clampedRatio };
}