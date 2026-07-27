import React, { useCallback, useEffect, useRef, useState } from 'react';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import {
  Box,
  Chip,
  Divider,
  IconButton,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material';
import type { AgentWorkflowCatalogResponse } from '../../lib/api';
import type { AgentWorkflowBuilderState, BuilderEdgeState } from '../../lib/agent-workflow-builder';
import { clampPanelRatio, PANEL_RATIOS } from '../../lib/panel-ratio';
import type { BuilderSelection, BuilderValidationIssue } from './types';
import WorkflowBuilderCanvas from './WorkflowBuilderCanvas';
import BuilderResizeHandle from './BuilderResizeHandle';

const edgeLabel = (edge: BuilderEdgeState) => {
  if (edge.conditional) {
    const routes = Object.entries(edge.routes || {})
      .map(([route, target]) => `${route}:${target}`)
      .join(', ');
    return `${edge.from} routes ${routes || '(empty)'}`;
  }
  return `${edge.from} -> ${edge.to}`;
};

export default function BuilderGraphEditor({
  catalog,
  state,
  selection,
  validationIssues,
  disabled,
  onSelectionChange,
  onConnectNodes,
  onRemoveEdge,
  onRemoveNode,
  onNodePositionChange,
  onAddNodeAt,
  onRequestAddPrevious,
  onRequestAddNext,
  onUpdateNote,
  onRemoveNote,
  onPositionsChange,
  graphElementsRatio,
  graphElementsCollapsed,
  onGraphElementsRatioChange,
  onGraphElementsCollapsedChange,
}: {
  catalog: AgentWorkflowCatalogResponse;
  state: AgentWorkflowBuilderState;
  selection: BuilderSelection;
  validationIssues: BuilderValidationIssue[];
  disabled?: boolean;
  onSelectionChange: (selection: BuilderSelection) => void;
  onConnectNodes: (source: string, target: string, route?: string) => void;
  onRemoveEdge: (index: number, route?: string) => void;
  onRemoveNode: (id: string) => void;
  onNodePositionChange: (nodeId: string, position: { x: number; y: number }) => void;
  onAddNodeAt: (nodeType: string, position: { x: number; y: number }) => void;
  onRequestAddPrevious: (target: string) => void;
  onRequestAddNext: (source: string, route?: string) => void;
  onUpdateNote: (id: string, position: { x: number; y: number }) => void;
  onRemoveNote: (id: string) => void;
  onPositionsChange: (positions: Record<string, { x: number; y: number }>) => void;
  graphElementsRatio: number;
  graphElementsCollapsed: boolean;
  onGraphElementsRatioChange: (ratio: number) => void;
  onGraphElementsCollapsedChange: (collapsed: boolean) => void;
}) {
  const editorRef = useRef<HTMLDivElement | null>(null);
  const [editorHeight, setEditorHeight] = useState(800);
  const graphElementsResolvedRatio = clampPanelRatio(graphElementsRatio, PANEL_RATIOS.graphElements);
  const graphElementsHeight = Math.round(editorHeight * graphElementsResolvedRatio);
  const graphElementsMinHeight = Math.max(72, Math.round(editorHeight * PANEL_RATIOS.graphElements.min));
  const graphElementsMaxHeight = Math.max(graphElementsMinHeight, Math.round(editorHeight * PANEL_RATIOS.graphElements.max));

  useEffect(() => {
    const element = editorRef.current;
    if (!element || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(([entry]) => setEditorHeight(entry.contentRect.height));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (graphElementsRatio !== graphElementsResolvedRatio) onGraphElementsRatioChange(graphElementsResolvedRatio);
  }, [graphElementsRatio, graphElementsResolvedRatio, onGraphElementsRatioChange]);

  const issueCountForSelection = useCallback((targetSelection: BuilderSelection) => (
    validationIssues.filter((issue) => {
      if (!issue.selection || !targetSelection || issue.selection.kind !== targetSelection.kind) return false;
      if (issue.selection.kind === 'node' && targetSelection.kind === 'node') {
        return issue.selection.nodeId === targetSelection.nodeId;
      }
      if (issue.selection.kind === 'edge' && targetSelection.kind === 'edge') {
        return issue.selection.edgeIndex === targetSelection.edgeIndex;
      }
      return false;
    }).length
  ), [validationIssues]);

  return (
    <Box
      ref={editorRef}
      sx={{
        display: 'grid',
        gridTemplateRows: `minmax(48px, 1fr) 8px ${graphElementsCollapsed ? 42 : graphElementsHeight}px`,
        minHeight: 400,
        height: '100%',
        flex: '1 1 400px',
        overflow: 'hidden',
        bgcolor: 'background.default',
      }}
    >
      <Box sx={{ minHeight: 48, overflow: 'hidden' }}>
        <WorkflowBuilderCanvas
          catalog={catalog}
          state={state}
          selection={selection}
          issues={validationIssues}
          disabled={disabled}
          onSelectionChange={onSelectionChange}
          onConnectNodes={onConnectNodes}
          onRemoveEdge={onRemoveEdge}
          onRemoveNode={onRemoveNode}
          onNodePositionChange={onNodePositionChange}
          onAddNodeAt={onAddNodeAt}
          onRequestAddPrevious={onRequestAddPrevious}
          onRequestAddNext={onRequestAddNext}
          onUpdateNote={onUpdateNote}
          onRemoveNote={onRemoveNote}
          onPositionsChange={onPositionsChange}
        />
      </Box>
      <BuilderResizeHandle
        orientation="horizontal"
        value={graphElementsHeight}
        min={graphElementsMinHeight}
        max={graphElementsMaxHeight}
        defaultValue={Math.round(editorHeight * PANEL_RATIOS.graphElements.default)}
        direction={-1}
        label="Resize Graph Elements panel"
        onChange={(height) => {
          onGraphElementsCollapsedChange(false);
          onGraphElementsRatioChange(clampPanelRatio(height / Math.max(1, editorHeight), PANEL_RATIOS.graphElements));
        }}
      />
      <Box sx={{ minWidth: 0, overflow: 'hidden', bgcolor: 'background.paper' }}>
        <Box sx={{ height: 40, px: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: graphElementsCollapsed ? 0 : 1, borderColor: 'divider' }}>
          <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 0.75, fontWeight: 700 }}>
            <AccountTreeIcon fontSize="small" /> Graph Elements
            <Chip size="small" variant="outlined" label={`${state.nodes.length} nodes · ${state.edges.length} edges`} sx={{ height: 21 }} />
          </Typography>
          <Tooltip title={graphElementsCollapsed ? 'Expand Graph Elements' : 'Collapse Graph Elements'}>
            <IconButton
              size="small"
              aria-label={graphElementsCollapsed ? 'Expand Graph Elements' : 'Collapse Graph Elements'}
              onClick={() => onGraphElementsCollapsedChange(!graphElementsCollapsed)}
            >
              {graphElementsCollapsed ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          </Tooltip>
        </Box>
        {!graphElementsCollapsed ? (
          <Box sx={{ height: 'calc(100% - 40px)', overflowY: 'auto', overflowX: 'hidden', px: 1, py: 0.75 }}>
          <Typography variant="caption" sx={{ display: 'block', mb: 0.5, fontWeight: 700, color: 'text.secondary', textTransform: 'uppercase' }}>
            Nodes
          </Typography>
          <Stack direction="row" spacing={0.75} sx={{ flexWrap: 'wrap', alignItems: 'center', rowGap: 0.75 }}>
            {state.nodes.map((node) => (
              <Box key={node.id} sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.35, maxWidth: '100%' }}>
                <Chip
                  clickable
                  color={selection?.kind === 'node' && selection.nodeId === node.id ? 'primary' : 'default'}
                  variant={selection?.kind === 'node' && selection.nodeId === node.id ? 'filled' : 'outlined'}
                  label={`${node.id} · ${node.type}`}
                  onClick={() => onSelectionChange({ kind: 'node', nodeId: node.id })}
                  sx={{ maxWidth: '100%' }}
                />
                {issueCountForSelection({ kind: 'node', nodeId: node.id }) > 0 ? (
                  <Chip size="small" color="error" label={issueCountForSelection({ kind: 'node', nodeId: node.id })} />
                ) : null}
              </Box>
            ))}
          </Stack>
          <Divider sx={{ my: 1 }} />
          <Typography variant="caption" sx={{ display: 'block', mb: 0.5, fontWeight: 700, color: 'text.secondary', textTransform: 'uppercase' }}>
            Edges
          </Typography>
          <Stack direction="row" spacing={0.75} sx={{ flexWrap: 'wrap', alignItems: 'center', rowGap: 0.75 }}>
            {state.edges.map((edge, index) => (
              <Box key={`${edge.from}-${edge.to || 'routes'}-${index}`} sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.35, maxWidth: '100%' }}>
                <Chip
                  clickable
                  color={selection?.kind === 'edge' && selection.edgeIndex === index ? 'primary' : 'default'}
                  variant={selection?.kind === 'edge' && selection.edgeIndex === index ? 'filled' : 'outlined'}
                  label={edgeLabel(edge)}
                  onClick={() => onSelectionChange({ kind: 'edge', edgeIndex: index })}
                  sx={{ maxWidth: '100%' }}
                />
                {issueCountForSelection({ kind: 'edge', edgeIndex: index }) > 0 ? (
                  <Chip size="small" color="error" label={issueCountForSelection({ kind: 'edge', edgeIndex: index })} />
                ) : null}
              </Box>
            ))}
          </Stack>
          </Box>
        ) : null}
      </Box>
    </Box>
  );
}
