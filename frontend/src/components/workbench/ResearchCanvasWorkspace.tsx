import React, { useCallback, useEffect, useRef } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import DashboardOutlinedIcon from '@mui/icons-material/DashboardOutlined';
import { getThreadCanvas } from '../../lib/canvas-api';
import type { DocumentCanvasCitationTarget } from '../../lib/canvas-spec';
import { buildCanvasWorkspaceDeepLink } from '../../lib/workspace-deep-link.ts';
import CanvasDocument from '../canvas/CanvasDocument';
import NestedInstanceTabs from './NestedInstanceTabs';
import InstanceTabCloseLabel from './InstanceTabCloseLabel';
import WorkspaceEmptyState from './WorkspaceEmptyState';
import type { CanvasRunTab } from './useCanvasTabs';

const copyTextToClipboard = async (text: string) => {
  try {
    await navigator.clipboard?.writeText(text);
  } catch {
    window.prompt('Copy link', text);
  }
};

export default function ResearchCanvasWorkspace({
  threadId,
  projectId = null,
  canvasTabs,
  activeCanvasId,
  onActiveCanvasChange,
  onCloseCanvas,
  onOpenDocumentCitation,
  onCanvasTabUpdate,
  refreshVersion = 0,
  darkMode = false,
}: {
  threadId: string | null;
  projectId?: string | null;
  canvasTabs: CanvasRunTab[];
  activeCanvasId: string | null;
  onActiveCanvasChange: (canvasId: string) => void;
  onCloseCanvas: (canvasId: string) => void;
  onOpenDocumentCitation?: (target: DocumentCanvasCitationTarget) => void;
  onCanvasTabUpdate: (canvasId: string, patch: Partial<CanvasRunTab>) => void;
  refreshVersion?: number;
  darkMode?: boolean;
}) {
  const hydrateTab = useCallback(async (canvasId: string) => {
    if (!threadId) return;
    onCanvasTabUpdate(canvasId, { loading: true, error: null });
    try {
      const record = await getThreadCanvas(threadId, canvasId);
      onCanvasTabUpdate(canvasId, {
        title: record.title,
        record,
        loading: false,
        error: null,
      });
    } catch (err) {
      onCanvasTabUpdate(canvasId, {
        loading: false,
        error: err instanceof Error ? err.message : 'Unable to load canvas',
      });
    }
  }, [onCanvasTabUpdate, threadId]);

  const canvasTabsRef = useRef(canvasTabs);
  canvasTabsRef.current = canvasTabs;

  useEffect(() => {
    if (!threadId) return;
    for (const tab of canvasTabs) {
      if (!tab.record && !tab.loading) {
        void hydrateTab(tab.id);
      }
    }
  }, [canvasTabs, hydrateTab, threadId]);

  useEffect(() => {
    if (!threadId || refreshVersion <= 0) return;
    for (const tab of canvasTabsRef.current) {
      void hydrateTab(tab.id);
    }
  }, [hydrateTab, refreshVersion, threadId]);

  const activeTab = canvasTabs.find((tab) => tab.id === activeCanvasId) || canvasTabs[0] || null;

  if (!threadId) {
    return (
    <Box sx={{ height: '100%', display: 'grid', placeItems: 'center', p: 3, bgcolor: darkMode ? '#222' : 'grey.50', color: 'text.secondary' }}>
        <Typography>Open a thread to view research canvases.</Typography>
      </Box>
    );
  }

  const nestedTabs = canvasTabs.map((tab) => ({
    id: tab.id,
    label: (
      <InstanceTabCloseLabel
        title={tab.title}
        copyLinkTooltip="Copy canvas link"
        onCopyLink={threadId ? async () => {
          const url = buildCanvasWorkspaceDeepLink({
            threadId,
            projectId,
            canvasId: tab.id,
          });
          await copyTextToClipboard(url);
        } : undefined}
        onClose={() => onCloseCanvas(tab.id)}
        closeTooltip="Close canvas"
      />
    ),
  }));

  return (
    <Box sx={{ height: '100%', minHeight: 0, display: 'grid', bgcolor: darkMode ? '#222' : 'grey.50', gridTemplateRows: canvasTabs.length > 0 ? 'auto minmax(0, 1fr)' : 'minmax(0, 1fr)' }}>
      {canvasTabs.length > 0 ? (
        <NestedInstanceTabs
          tabs={nestedTabs}
          activeId={activeTab?.id || null}
          onActiveChange={onActiveCanvasChange}
          ariaLabel="Open canvases"
        />
      ) : null}
      <Box sx={{ minHeight: 0, overflow: 'auto', bgcolor: darkMode ? '#222' : 'grey.50' }}>
        {activeTab?.error ? (
          <Typography color="error" sx={{ p: 2 }}>{activeTab.error}</Typography>
        ) : activeTab?.record ? (
          <CanvasDocument
            spec={activeTab.record.spec}
            createdAt={activeTab.record.created_at}
            onOpenDocumentCitation={onOpenDocumentCitation}
          />
        ) : activeTab?.loading ? (
          <Box sx={{ display: 'grid', placeItems: 'center', height: '100%' }}><CircularProgress size={24} /></Box>
        ) : (
          <WorkspaceEmptyState
            icon={<DashboardOutlinedIcon sx={{ fontSize: 48, opacity: 0.4 }} />}
            title="No canvas open"
            description="Open a canvas from chat or deep research to inspect it here. Open canvases stay in the tab strip until you close them."
            maxWidth={420}
            darkMode={darkMode}
          />
        )}
      </Box>
    </Box>
  );
}
