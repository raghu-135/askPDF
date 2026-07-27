import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Box, type SxProps, type Theme } from '@mui/material';
import {
  DEFAULT_WORKBENCH_LAYOUT,
  normalizeWorkbenchLayout,
  resizeWorkbenchRatio,
  resolveWorkbenchPlacement,
  type ResolvedWorkbenchPlacement,
  type WorkbenchLayoutState,
} from '../../lib/workbench-layout';
import useStoredLayoutState from './useStoredLayoutState';

export const useWorkbenchLayout = (
  storageKey: string,
  defaults: Partial<WorkbenchLayoutState> = {},
) => {
  const fallback = useMemo(
    () => normalizeWorkbenchLayout({ ...DEFAULT_WORKBENCH_LAYOUT, ...defaults }),
    [defaults.bottomRatio, defaults.placement, defaults.sideRatio, defaults.visible],
  );
  return useStoredLayoutState(storageKey, fallback, normalizeWorkbenchLayout);
};

export default function WorkbenchShell({
  primaryToolbar,
  primaryTabs,
  primaryContent,
  secondaryHeader,
  secondaryControls,
  secondaryContent,
  layout,
  onLayoutChange,
  onResolvedPlacementChange,
  onResizingChange,
  autoSideMinWidth = 1100,
  hardSideMinWidth = 720,
  dividerSize = 8,
  secondaryLabel = 'Secondary panel',
  sx,
}: {
  primaryToolbar?: React.ReactNode;
  primaryTabs?: React.ReactNode;
  primaryContent: React.ReactNode;
  secondaryHeader?: React.ReactNode;
  secondaryControls?: React.ReactNode;
  secondaryContent: React.ReactNode;
  layout: WorkbenchLayoutState;
  onLayoutChange: (layout: WorkbenchLayoutState) => void;
  onResolvedPlacementChange?: (placement: ResolvedWorkbenchPlacement) => void;
  onResizingChange?: (resizing: boolean) => void;
  autoSideMinWidth?: number;
  hardSideMinWidth?: number;
  dividerSize?: number;
  secondaryLabel?: string;
  sx?: SxProps<Theme>;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [resizing, setResizing] = useState(false);
  const resolvedPlacement = resolveWorkbenchPlacement(
    layout.placement,
    containerSize.width,
    autoSideMinWidth,
    hardSideMinWidth,
  );
  const isBottom = resolvedPlacement === 'bottom';

  useEffect(() => {
    onResolvedPlacementChange?.(resolvedPlacement);
  }, [onResolvedPlacementChange, resolvedPlacement]);

  useEffect(() => {
    const element = containerRef.current;
    if (!element || typeof ResizeObserver === 'undefined') return;
    const update = () => {
      const rect = element.getBoundingClientRect();
      setContainerSize({ width: rect.width, height: rect.height });
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const updateFromPointer = useCallback((clientX: number, clientY: number) => {
    const element = containerRef.current;
    if (!element) return;
    const rect = element.getBoundingClientRect();
    const ratio = resizeWorkbenchRatio({
      placement: resolvedPlacement,
      clientX,
      clientY,
      left: rect.left,
      right: rect.right,
      top: rect.top,
      bottom: rect.bottom,
    });
    onLayoutChange(normalizeWorkbenchLayout({
      ...layout,
      ...(isBottom ? { bottomRatio: ratio } : { sideRatio: ratio }),
    }));
  }, [isBottom, layout, onLayoutChange, resolvedPlacement]);

  const finishResize = useCallback(() => {
    setResizing(false);
    onResizingChange?.(false);
  }, [onResizingChange]);

  const handleKeyDown = useCallback((event: React.KeyboardEvent<HTMLDivElement>) => {
    const delta = event.shiftKey ? 0.05 : 0.02;
    let next: WorkbenchLayoutState | null = null;
    if (isBottom && (event.key === 'ArrowUp' || event.key === 'ArrowDown')) {
      next = normalizeWorkbenchLayout({
        ...layout,
        bottomRatio: layout.bottomRatio + (event.key === 'ArrowUp' ? delta : -delta),
      });
    } else if (!isBottom && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) {
      const growsOnRight = resolvedPlacement === 'right'
        ? event.key === 'ArrowLeft'
        : event.key === 'ArrowRight';
      next = normalizeWorkbenchLayout({
        ...layout,
        sideRatio: layout.sideRatio + (growsOnRight ? delta : -delta),
      });
    }
    if (next) {
      event.preventDefault();
      onLayoutChange(next);
    }
  }, [isBottom, layout, onLayoutChange, resolvedPlacement]);

  const primary = (
    <Box sx={{ height: '100%', minHeight: 0, minWidth: 0, display: 'grid', gridTemplateRows: `${primaryToolbar ? 'auto ' : ''}${primaryTabs ? 'auto ' : ''}minmax(0, 1fr)`, overflow: 'hidden' }}>
      {primaryToolbar}
      {primaryTabs}
      <Box sx={{ minHeight: 0, minWidth: 0, overflow: 'hidden' }}>{primaryContent}</Box>
    </Box>
  );
  const secondary = (
    <Box sx={{ height: '100%', minHeight: 0, minWidth: 0, display: 'grid', gridTemplateRows: `${secondaryHeader ? 'auto ' : ''}${secondaryControls ? 'auto ' : ''}minmax(0, 1fr)`, overflow: 'hidden' }}>
      {secondaryHeader}
      {secondaryControls}
      <Box sx={{ minHeight: 0, minWidth: 0, overflow: 'hidden' }}>{secondaryContent}</Box>
    </Box>
  );

  return (
    <Box
      ref={containerRef}
      sx={{
        height: '100%',
        minHeight: 0,
        minWidth: 0,
        display: 'flex',
        flexDirection: isBottom ? 'column' : 'row',
        overflow: 'hidden',
        ...sx,
      }}
    >
      <Box sx={{ order: 0, flex: '1 1 auto', minHeight: 0, minWidth: 0, overflow: 'hidden' }}>
        {primary}
      </Box>
      <Box
        role="separator"
        tabIndex={layout.visible ? 0 : -1}
        aria-label={`Resize ${secondaryLabel}`}
        aria-orientation={isBottom ? 'horizontal' : 'vertical'}
        onKeyDown={handleKeyDown}
        onPointerDown={(event) => {
          if (!layout.visible) return;
          event.preventDefault();
          event.currentTarget.setPointerCapture(event.pointerId);
          setResizing(true);
          onResizingChange?.(true);
        }}
        onPointerMove={(event) => resizing && updateFromPointer(event.clientX, event.clientY)}
        onPointerUp={(event) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
          finishResize();
        }}
        onPointerCancel={finishResize}
        sx={{
          order: resolvedPlacement === 'left' ? -1 : 1,
          display: layout.visible ? 'flex' : 'none',
          flex: `0 0 ${dividerSize}px`,
          cursor: isBottom ? 'row-resize' : 'col-resize',
          alignItems: 'center',
          justifyContent: 'center',
          touchAction: 'none',
          zIndex: 5,
          '&::after': {
            content: '""',
            width: isBottom ? '100%' : 2,
            height: isBottom ? 2 : '100%',
            bgcolor: resizing ? 'primary.main' : 'divider',
          },
          '&:hover::after, &:focus-visible::after': { bgcolor: 'primary.main' },
        }}
      />
      <Box
        aria-label={secondaryLabel}
        sx={{
          order: resolvedPlacement === 'left' ? -2 : 2,
          display: layout.visible ? 'block' : 'none',
          flex: `0 0 ${isBottom ? layout.bottomRatio * 100 : layout.sideRatio * 100}%`,
          minWidth: 0,
          minHeight: 0,
          overflow: 'hidden',
          bgcolor: 'background.paper',
        }}
      >
        {secondary}
      </Box>
      {resizing && <Box sx={{ position: 'fixed', inset: 0, zIndex: 9999, cursor: isBottom ? 'row-resize' : 'col-resize', userSelect: 'none' }} />}
    </Box>
  );
}
