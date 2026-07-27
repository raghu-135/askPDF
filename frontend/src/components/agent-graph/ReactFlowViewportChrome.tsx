import React from 'react';
import { useTheme, type SxProps, type Theme } from '@mui/material/styles';
import { Controls, MiniMap, type PanelPosition } from '@xyflow/react';

export const reactFlowChromeSx = (
  theme: Theme,
  canvasBg: string,
): SxProps<Theme> => ({
  '& .react-flow__controls': {
    overflow: 'hidden',
    border: 1,
    borderColor: 'divider',
    borderRadius: 1,
    boxShadow: theme.shadows[2],
    bgcolor: canvasBg,
  },
  '& .react-flow__controls-button': {
    width: 30,
    height: 30,
    bgcolor: canvasBg,
    color: 'text.primary',
    borderBottom: 1,
    borderBottomColor: 'divider',
  },
  '& .react-flow__controls-button:hover': {
    bgcolor: 'action.hover',
  },
  '& .react-flow__controls-button svg': {
    fill: 'currentColor',
  },
  '& .react-flow__minimap': {
    overflow: 'hidden',
    border: 1,
    borderColor: 'divider',
    borderRadius: 1,
    boxShadow: theme.shadows[2],
    bgcolor: 'background.paper',
  },
  '& .react-flow__minimap-mask': {
    fill: theme.palette.action.disabledBackground,
  },
});

export default function ReactFlowViewportChrome({
  showControls = true,
  controlsPosition = 'bottom-left',
  showInteractive = true,
  showMiniMap = true,
  miniMapPosition = 'top-left',
  pannable = true,
  zoomable = true,
}: {
  showControls?: boolean;
  controlsPosition?: PanelPosition;
  showInteractive?: boolean;
  showMiniMap?: boolean;
  miniMapPosition?: PanelPosition;
  pannable?: boolean;
  zoomable?: boolean;
}) {
  const theme = useTheme();
  const nodeColor = theme.palette.mode === 'dark'
    ? theme.palette.grey[700]
    : theme.palette.grey[300];
  const nodeStrokeColor = theme.palette.primary.main;

  return (
    <>
      {showControls ? (
        <Controls position={controlsPosition} showInteractive={showInteractive} />
      ) : null}
      {showMiniMap ? (
        <MiniMap
          position={miniMapPosition}
          pannable={pannable}
          zoomable={zoomable}
          nodeColor={nodeColor}
          nodeStrokeColor={nodeStrokeColor}
          maskColor={theme.palette.action.disabledBackground}
        />
      ) : null}
    </>
  );
}
