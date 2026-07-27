import React from 'react';
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Divider,
  IconButton,
  Tooltip,
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import type { ResolvedWorkbenchPlacement } from '../../lib/workbench-layout';
import BuilderResizeHandle from './BuilderResizeHandle';

export type BuilderUtilityLayout = {
  palettePercent: number;
  inspectorCollapsed: boolean;
};

export default function BuilderUtilityPanel({
  placement,
  layout,
  utilityRailRef,
  inspector,
  palette,
  stats,
  onPalettePercentChange,
  onInspectorCollapsedChange,
}: {
  placement: ResolvedWorkbenchPlacement;
  layout: BuilderUtilityLayout;
  utilityRailRef: React.RefObject<HTMLDivElement | null>;
  inspector: React.ReactNode;
  palette: React.ReactNode;
  stats: React.ReactNode;
  onPalettePercentChange: (palettePercent: number) => void;
  onInspectorCollapsedChange: (collapsed: boolean) => void;
}) {
  if (placement === 'bottom') {
    return (
      <Box sx={{ height: '100%', overflow: 'auto', p: 1 }}>
        <Accordion defaultExpanded disableGutters>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}><Typography variant="subtitle2">Inspector</Typography></AccordionSummary>
          <AccordionDetails>{inspector}</AccordionDetails>
        </Accordion>
        <Accordion defaultExpanded disableGutters>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}><Typography variant="subtitle2">Node Palette</Typography></AccordionSummary>
          <AccordionDetails>{palette}</AccordionDetails>
        </Accordion>
      </Box>
    );
  }

  return (
    <Box ref={utilityRailRef} sx={{ height: '100%', minHeight: 0, display: 'grid', gridTemplateRows: layout.inspectorCollapsed ? '44px minmax(180px, 1fr)' : `minmax(240px, ${100 - layout.palettePercent}fr) 8px minmax(180px, ${layout.palettePercent}fr)` }}>
      <Box sx={{ minHeight: 0, overflow: 'hidden', borderBottom: layout.inspectorCollapsed ? 1 : 0, borderColor: 'divider' }}>
        <Box sx={{ height: 44, px: 1.5, display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: layout.inspectorCollapsed ? 0 : 1, borderColor: 'divider' }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>Inspector</Typography>
          <Tooltip title={layout.inspectorCollapsed ? 'Expand Inspector' : 'Collapse Inspector'}>
            <IconButton size="small" onClick={() => onInspectorCollapsedChange(!layout.inspectorCollapsed)}>
              {layout.inspectorCollapsed ? <ExpandMoreIcon /> : <ExpandLessIcon />}
            </IconButton>
          </Tooltip>
        </Box>
        {!layout.inspectorCollapsed && (
          <Box sx={{ height: 'calc(100% - 44px)', overflow: 'auto', p: 1.5 }}>
            {inspector}
            <Divider sx={{ my: 1.5 }} />
            {stats}
          </Box>
        )}
      </Box>
      {!layout.inspectorCollapsed && (
        <BuilderResizeHandle
          orientation="horizontal"
          value={layout.palettePercent}
          min={25}
          max={65}
          defaultValue={40}
          step={2}
          direction={-1}
          label="Resize Inspector and Node Palette"
          getDragScale={() => 100 / Math.max(1, utilityRailRef.current?.clientHeight || 700)}
          onChange={onPalettePercentChange}
        />
      )}
      <Box sx={{ minHeight: 0, overflow: 'auto', p: 1.5 }}>
        {palette}
      </Box>
    </Box>
  );
}
