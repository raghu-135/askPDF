import React, { useEffect, useState } from 'react';
import {
  Box,
  Tab,
  Tabs,
} from '@mui/material';
import type { ResolvedWorkbenchPlacement } from '../../lib/workbench-layout';

type UtilityTab = 'inspector' | 'palette';

export default function BuilderUtilityPanel({
  placement,
  utilityRailRef,
  selectionKey,
  inspector,
  palette,
}: {
  placement: ResolvedWorkbenchPlacement;
  utilityRailRef: React.RefObject<HTMLDivElement | null>;
  selectionKey?: string | null;
  inspector: React.ReactNode;
  palette: React.ReactNode;
}) {
  const [activeTab, setActiveTab] = useState<UtilityTab>(selectionKey ? 'inspector' : 'palette');

  useEffect(() => {
    setActiveTab(selectionKey ? 'inspector' : 'palette');
  }, [selectionKey]);

  return (
    <Box
      ref={utilityRailRef}
      sx={{
        height: '100%',
        minHeight: 0,
        display: 'grid',
        gridTemplateRows: 'auto minmax(0, 1fr)',
        bgcolor: 'background.paper',
      }}
    >
      <Tabs
        value={activeTab}
        onChange={(_, next: UtilityTab) => setActiveTab(next)}
        variant={placement === 'bottom' ? 'standard' : 'fullWidth'}
        aria-label="Builder utility panel"
        sx={{
          minHeight: 38,
          borderBottom: 1,
          borderColor: 'divider',
          '& .MuiTab-root': {
            minHeight: 38,
            py: 0,
            textTransform: 'none',
            fontSize: '0.82rem',
          },
        }}
      >
        <Tab value="inspector" label="Inspector" />
        <Tab value="palette" label="Palette" />
      </Tabs>
      <Box sx={{ minHeight: 0, overflow: 'auto', p: 1 }}>
        {activeTab === 'inspector' ? inspector : palette}
      </Box>
    </Box>
  );
}
