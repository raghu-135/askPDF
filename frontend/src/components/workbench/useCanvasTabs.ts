import { useCallback } from 'react';
import type { CanvasRef } from '../../lib/canvas-spec';
import type { ThreadCanvasRecord } from '../../lib/canvas-spec';
import { upsertInstanceTab } from '../../lib/instance-tabs.ts';
import useInstanceTabs from './useInstanceTabs';

export type CanvasRunTab = {
  id: string;
  title: string;
  record?: ThreadCanvasRecord;
  loading?: boolean;
  error?: string | null;
};

const mergeCanvasTab = (tab: CanvasRunTab | undefined, nextTab: CanvasRunTab): CanvasRunTab => {
  if (!tab) return nextTab;
  const record = nextTab.record ?? tab.record;
  return {
    ...tab,
    ...nextTab,
    ...(record === undefined ? {} : { record }),
  };
};

export default function useCanvasTabs() {
  const {
    tabs: canvasTabs,
    setTabs: setCanvasTabs,
    activeId: activeCanvasId,
    setActiveId: setActiveCanvasId,
    close: closeCanvas,
    clear: clearCanvasTabs,
  } = useInstanceTabs<CanvasRunTab>();

  const openCanvas = useCallback((
    canvas: CanvasRef,
    options?: { activate?: boolean },
  ) => {
    setCanvasTabs((current) => upsertInstanceTab(
      current,
      { id: canvas.id, title: canvas.title },
      { merge: mergeCanvasTab },
    ));
    if (options?.activate !== false) setActiveCanvasId(canvas.id);
  }, [setActiveCanvasId, setCanvasTabs]);

  const updateCanvasTab = useCallback((canvasId: string, patch: Partial<CanvasRunTab>) => {
    setCanvasTabs((current) => current.map((tab) => (
      tab.id === canvasId ? mergeCanvasTab(tab, { ...tab, ...patch, id: tab.id }) : tab
    )));
  }, [setCanvasTabs]);

  return {
    canvasTabs,
    setCanvasTabs,
    activeCanvasId,
    setActiveCanvasId,
    openCanvas,
    closeCanvas,
    clearCanvasTabs,
    updateCanvasTab,
  };
}