import React, { useCallback, useEffect, useState } from 'react';
import { Box, CircularProgress, Tab, Tabs, Typography } from '@mui/material';
import DashboardOutlinedIcon from '@mui/icons-material/DashboardOutlined';
import { listThreadCanvases } from '../../lib/canvas-api';
import type { DocumentCanvasCitationTarget, ThreadCanvasRecord } from '../../lib/canvas-spec';
import CanvasDocument from '../canvas/CanvasDocument';

export default function ResearchCanvasWorkspace({
  threadId,
  activeCanvasId,
  onActiveCanvasChange,
  onOpenDocumentCitation,
  refreshVersion = 0,
}: {
  threadId: string | null;
  activeCanvasId: string | null;
  onActiveCanvasChange: (canvasId: string) => void;
  onOpenDocumentCitation?: (target: DocumentCanvasCitationTarget) => void;
  refreshVersion?: number;
}) {
  const [canvases, setCanvases] = useState<ThreadCanvasRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!threadId) {
      setCanvases([]);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await listThreadCanvases(threadId);
      setCanvases(response.canvases);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load canvases');
    } finally {
      setLoading(false);
    }
  }, [threadId]);

  useEffect(() => {
    void load();
  }, [load, refreshVersion]);

  useEffect(() => {
    if (!activeCanvasId && canvases[0]) onActiveCanvasChange(canvases[0].id);
  }, [activeCanvasId, canvases, onActiveCanvasChange]);

  const active = canvases.find((canvas) => canvas.id === activeCanvasId) || canvases[0] || null;

  if (!threadId) {
    return (
      <Box sx={{ height: '100%', display: 'grid', placeItems: 'center', p: 3, color: 'text.secondary' }}>
        <Typography>Open a thread to view research canvases.</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', minHeight: 0, display: 'grid', gridTemplateRows: 'auto minmax(0, 1fr)' }}>
      <Tabs
        value={active ? canvases.findIndex((canvas) => canvas.id === active.id) : false}
        onChange={(_, index) => canvases[index] && onActiveCanvasChange(canvases[index].id)}
        variant="scrollable"
        scrollButtons="auto"
        aria-label="Research canvases"
        sx={{ minHeight: 38, borderBottom: 1, borderColor: 'divider' }}
      >
        {canvases.map((canvas) => (
          <Tab key={canvas.id} label={canvas.title} sx={{ textTransform: 'none', minHeight: 38 }} />
        ))}
      </Tabs>
      <Box sx={{ minHeight: 0, overflow: 'auto' }}>
        {loading ? (
          <Box sx={{ display: 'grid', placeItems: 'center', height: '100%' }}><CircularProgress size={24} /></Box>
        ) : error ? (
          <Typography color="error" sx={{ p: 2 }}>{error}</Typography>
        ) : active ? (
          <CanvasDocument spec={active.spec} onOpenDocumentCitation={onOpenDocumentCitation} />
        ) : (
          <Box sx={{ height: '100%', display: 'grid', placeItems: 'center', p: 4, color: 'text.secondary', textAlign: 'center' }}>
            <Box>
              <DashboardOutlinedIcon sx={{ fontSize: 42, opacity: 0.45 }} />
              <Typography variant="h6">No canvas yet</Typography>
              <Typography variant="body2">Structured research canvases opened from chat will appear here.</Typography>
            </Box>
          </Box>
        )}
      </Box>
    </Box>
  );
}
