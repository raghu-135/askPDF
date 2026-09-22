import React, { useCallback, useEffect, useState } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import DashboardOutlinedIcon from '@mui/icons-material/DashboardOutlined';
import { listThreadCanvases } from '../../lib/canvas-api';
import type { DocumentCanvasCitationTarget, ThreadCanvasRecord } from '../../lib/canvas-spec';
import CanvasDocument from '../canvas/CanvasDocument';
import NestedInstanceTabs from './NestedInstanceTabs';
import WorkspaceEmptyState from './WorkspaceEmptyState';

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

  const nestedTabs = canvases.map((canvas) => ({
    id: canvas.id,
    label: canvas.title,
  }));

  return (
    <Box sx={{ height: '100%', minHeight: 0, display: 'grid', gridTemplateRows: canvases.length > 1 ? 'auto minmax(0, 1fr)' : 'minmax(0, 1fr)' }}>
      {canvases.length > 1 ? (
        <NestedInstanceTabs
          tabs={nestedTabs}
          activeId={active?.id || null}
          onActiveChange={onActiveCanvasChange}
          ariaLabel="Research canvases"
        />
      ) : null}
      <Box sx={{ minHeight: 0, overflow: 'auto' }}>
        {loading ? (
          <Box sx={{ display: 'grid', placeItems: 'center', height: '100%' }}><CircularProgress size={24} /></Box>
        ) : error ? (
          <Typography color="error" sx={{ p: 2 }}>{error}</Typography>
        ) : active ? (
          <CanvasDocument
            spec={active.spec}
            createdAt={active.created_at}
            onOpenDocumentCitation={onOpenDocumentCitation}
          />
        ) : (
          <WorkspaceEmptyState
            icon={<DashboardOutlinedIcon sx={{ fontSize: 48, opacity: 0.4 }} />}
            title="No canvas yet"
            description="When chat publishes a structured research canvas, it will land here as a durable document you can jump back to from sources and citations."
            maxWidth={420}
          />
        )}
      </Box>
    </Box>
  );
}
