import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';
import { clampDecisionPanelRatio } from '../lib/conversation-ui-state';

type ResizablePanelRatioBounds = {
  min?: number;
  max?: number;
};

export function useResizablePanelRatio(
  rootRef: RefObject<HTMLElement | null>,
  defaultRatio = 0.3,
  bounds: ResizablePanelRatioBounds = {},
) {
  const { min, max } = bounds;
  const [ratio, setRatio] = useState(() => clampDecisionPanelRatio(defaultRatio, min, max));
  const [resizing, setResizing] = useState(false);
  const resizeRef = useRef({ startY: 0, startRatio: ratio });

  const onResizeStart = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    resizeRef.current = { startY: event.clientY, startRatio: ratio };
    setResizing(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  }, [ratio]);

  const handleResizeMove = useCallback((event: PointerEvent) => {
    const panelHeight = rootRef.current?.getBoundingClientRect().height || window.innerHeight;
    const deltaRatio = (resizeRef.current.startY - event.clientY) / panelHeight;
    setRatio(clampDecisionPanelRatio(resizeRef.current.startRatio + deltaRatio, min, max));
  }, [max, min, rootRef]);

  const handleResizeEnd = useCallback(() => setResizing(false), []);

  useEffect(() => {
    if (!resizing) return;
    document.body.style.cursor = 'ns-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('pointermove', handleResizeMove);
    document.addEventListener('pointerup', handleResizeEnd);
    document.addEventListener('pointercancel', handleResizeEnd);
    return () => {
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('pointermove', handleResizeMove);
      document.removeEventListener('pointerup', handleResizeEnd);
      document.removeEventListener('pointercancel', handleResizeEnd);
    };
  }, [handleResizeEnd, handleResizeMove, resizing]);

  return { ratio, resizing, onResizeStart };
}
