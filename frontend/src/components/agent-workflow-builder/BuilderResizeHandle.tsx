import React, { useEffect, useRef } from 'react';
import { Box } from '@mui/material';

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

export default function BuilderResizeHandle({
  orientation,
  value,
  min,
  max,
  defaultValue,
  onChange,
  label,
  direction = 1,
  step = 8,
  getDragScale,
  sx,
}: {
  orientation: 'vertical' | 'horizontal';
  value: number;
  min: number;
  max: number;
  defaultValue: number;
  onChange: (value: number) => void;
  label: string;
  direction?: 1 | -1;
  step?: number;
  getDragScale?: () => number;
  sx?: Record<string, any>;
}) {
  const drag = useRef<{ coordinate: number; value: number } | null>(null);
  useEffect(() => () => {
    if (typeof document !== 'undefined') document.body.style.userSelect = '';
  }, []);
  const coordinate = (event: React.PointerEvent) => orientation === 'vertical' ? event.clientX : event.clientY;
  const change = (next: number) => onChange(clamp(next, min, max));

  return (
    <Box
      role="separator"
      tabIndex={0}
      aria-label={label}
      aria-orientation={orientation}
      aria-valuemin={Math.round(min)}
      aria-valuemax={Math.round(max)}
      aria-valuenow={Math.round(value)}
      onDoubleClick={() => change(defaultValue)}
      onPointerDown={(event) => {
        drag.current = { coordinate: coordinate(event), value };
        event.currentTarget.setPointerCapture(event.pointerId);
        if (typeof document !== 'undefined') document.body.style.userSelect = 'none';
      }}
      onPointerMove={(event) => {
        if (!drag.current || !event.currentTarget.hasPointerCapture(event.pointerId)) return;
        const scale = getDragScale?.() || 1;
        change(drag.current.value + (coordinate(event) - drag.current.coordinate) * direction * scale);
      }}
      onPointerUp={(event) => {
        drag.current = null;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
        if (typeof document !== 'undefined') document.body.style.userSelect = '';
      }}
      onPointerCancel={() => {
        drag.current = null;
        if (typeof document !== 'undefined') document.body.style.userSelect = '';
      }}
      onKeyDown={(event) => {
        const negativeKey = orientation === 'vertical' ? 'ArrowLeft' : 'ArrowUp';
        const positiveKey = orientation === 'vertical' ? 'ArrowRight' : 'ArrowDown';
        if (event.key === 'Home') change(min);
        else if (event.key === 'End') change(max);
        else if (event.key === negativeKey) change(value - step * direction);
        else if (event.key === positiveKey) change(value + step * direction);
        else return;
        event.preventDefault();
      }}
      sx={{
        position: 'relative',
        flex: '0 0 auto',
        cursor: orientation === 'vertical' ? 'col-resize' : 'row-resize',
        outline: 'none',
        touchAction: 'none',
        bgcolor: 'transparent',
        '&::after': {
          content: '""',
          position: 'absolute',
          bgcolor: 'divider',
          borderRadius: 2,
          transition: 'background-color 120ms ease, width 120ms ease, height 120ms ease',
          ...(orientation === 'vertical'
            ? { width: 2, top: 8, bottom: 8, left: '50%', transform: 'translateX(-50%)' }
            : { height: 2, left: 8, right: 8, top: '50%', transform: 'translateY(-50%)' }),
        },
        '&:hover::after, &:focus-visible::after': {
          bgcolor: 'primary.main',
          ...(orientation === 'vertical' ? { width: 4 } : { height: 4 }),
        },
        ...sx,
      }}
    />
  );
}
