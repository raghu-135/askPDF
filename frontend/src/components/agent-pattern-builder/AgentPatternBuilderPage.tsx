import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  CircularProgress,
  CssBaseline,
  Divider,
  Typography,
} from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../../theme';
import {
  getInternalAgentPatternCatalog,
  validateAgentPatternSpec,
  type AgentPatternCatalogResponse,
  type AgentPatternValidationReport,
} from '../../lib/api';
import {
  assembleAgentPatternSpec,
  canAddNodeType,
  canConnectNodes,
  createHitlGateForTarget,
  createInitialBuilderState,
  getAllowedToolContractsForNode,
  getCanonicalNodeId,
  normalizeBuilderState,
  type AgentPatternBuilderState,
  type AgentPatternStarter,
  type BuilderEdgeState,
  type BuilderNodeState,
} from '../../lib/agent-pattern-builder';
import BuilderActionsBar from './BuilderActionsBar';
import BuilderGraphEditor from './BuilderGraphEditor';
import BuilderInspector from './BuilderInspector';
import BuilderNodePalette from './BuilderNodePalette';
import type { BuilderSelection } from './types';

const collectNodeToolIds = (nodes: BuilderNodeState[]) => (
  Array.from(new Set(nodes.flatMap((node) => node.tool_contract_ids || []))).sort()
);

const usePrefersDarkMode = () => {
  const [darkMode, setDarkMode] = useState(false);
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    setDarkMode(media.matches);
    const handler = (event: MediaQueryListEvent) => setDarkMode(event.matches);
    media.addEventListener('change', handler);
    return () => media.removeEventListener('change', handler);
  }, []);
  return darkMode;
};

export default function AgentPatternBuilderPage() {
  const darkMode = usePrefersDarkMode();
  const theme = useMemo(() => getTheme(darkMode), [darkMode]);
  const [catalog, setCatalog] = useState<AgentPatternCatalogResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [starter, setStarter] = useState<AgentPatternStarter>('router');
  const [builderState, setBuilderState] = useState<AgentPatternBuilderState | null>(null);
  const [selection, setSelection] = useState<BuilderSelection>(null);
  const [validation, setValidation] = useState<AgentPatternValidationReport | null>(null);
  const [validating, setValidating] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getInternalAgentPatternCatalog()
      .then((nextCatalog) => {
        if (cancelled) return;
        setCatalog(nextCatalog);
        setBuilderState(createInitialBuilderState(nextCatalog, starter));
        setError(null);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const updateState = useCallback((updater: (previous: AgentPatternBuilderState) => AgentPatternBuilderState) => {
    if (!catalog) return;
    setBuilderState((previous) => {
      if (!previous) return previous;
      const next = normalizeBuilderState(catalog, updater(previous));
      return {
        ...next,
        allowed_tool_ids: collectNodeToolIds(next.nodes),
      };
    });
    setValidation(null);
  }, [catalog]);

  const resetToStarter = useCallback((nextStarter = starter) => {
    if (!catalog) return;
    setBuilderState(createInitialBuilderState(catalog, nextStarter));
    setSelection(null);
    setValidation(null);
  }, [catalog, starter]);

  const handleStarterChange = (nextStarter: AgentPatternStarter) => {
    setStarter(nextStarter);
    resetToStarter(nextStarter);
  };

  const handleAddNodeType = (nodeType: string) => {
    if (!catalog || !builderState) return;
    const compatibility = canAddNodeType(catalog, builderState, nodeType);
    if (!compatibility.ok) return;
    const id = getCanonicalNodeId(nodeType, builderState.nodes.map((node) => node.id));
    const allowedTools = getAllowedToolContractsForNode(catalog, nodeType);
    const nextNode: BuilderNodeState = {
      id,
      type: nodeType,
      ...(allowedTools[0] ? { tool_contract_ids: [allowedTools[0].id] } : {}),
    };
    updateState((previous) => ({
      ...previous,
      nodes: [...previous.nodes, nextNode],
    }));
    setSelection({ kind: 'node', nodeId: id });
  };

  const handleUpdateNode = (nodeId: string, patch: Partial<BuilderNodeState>) => {
    updateState((previous) => ({
      ...previous,
      nodes: previous.nodes.map((node) => (node.id === nodeId ? { ...node, ...patch } : node)),
    }));
  };

  const handleRemoveNode = (nodeId: string) => {
    updateState((previous) => ({
      ...previous,
      nodes: previous.nodes.filter((node) => node.id !== nodeId),
      edges: previous.edges
        .map((edge) => {
          if (edge.from === nodeId || edge.to === nodeId) return null;
          if (!edge.routes) return edge;
          const routes = Object.fromEntries(Object.entries(edge.routes).filter(([, target]) => target !== nodeId));
          return { ...edge, routes };
        })
        .filter((edge): edge is BuilderEdgeState => Boolean(edge)),
    }));
    setSelection(null);
  };

  const handleAddEdge = (edge: BuilderEdgeState) => {
    if (!catalog || !builderState || !edge.to || !canConnectNodes(catalog, builderState, edge.from, edge.to).ok) return;
    const duplicate = builderState.edges.some((existing) => existing.from === edge.from && existing.to === edge.to && !existing.conditional);
    if (duplicate) return;
    updateState((previous) => ({
      ...previous,
      edges: [...previous.edges, edge],
    }));
    setSelection({ kind: 'edge', edgeIndex: builderState.edges.length });
  };

  const handleUpdateEdge = (edgeIndex: number, patch: Partial<BuilderEdgeState>) => {
    updateState((previous) => ({
      ...previous,
      edges: previous.edges.map((edge, index) => (index === edgeIndex ? { ...edge, ...patch } : edge)),
    }));
  };

  const handleRemoveEdge = (edgeIndex: number) => {
    updateState((previous) => ({
      ...previous,
      edges: previous.edges.filter((_, index) => index !== edgeIndex),
    }));
    setSelection(null);
  };

  const handleAddHitlGate = (targetNodeId: string) => {
    if (!catalog) return;
    updateState((previous) => createHitlGateForTarget(catalog, previous, targetNodeId));
  };

  const handleValidate = async () => {
    if (!builderState) return;
    try {
      setValidating(true);
      const report = await validateAgentPatternSpec(assembleAgentPatternSpec(builderState));
      setValidation(report);
    } catch (err) {
      setValidation({
        valid: false,
        errors: [err instanceof Error ? err.message : String(err)],
        warnings: [],
      });
    } finally {
      setValidating(false);
    }
  };

  const validationIssues = [
    ...(validation?.errors || []),
    ...(validation?.warnings || []).map((warning) => `Warning: ${warning}`),
  ];

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', display: 'grid', gridTemplateRows: 'auto minmax(0, 1fr)', bgcolor: 'background.default' }}>
        <BuilderActionsBar
          starter={starter}
          onStarterChange={handleStarterChange}
          onReset={() => resetToStarter()}
          onValidate={handleValidate}
          validating={validating}
          validation={validation}
        />
        {loading ? (
          <Box sx={{ display: 'grid', placeItems: 'center' }}>
            <CircularProgress />
          </Box>
        ) : error || !catalog || !builderState ? (
          <Box sx={{ p: 2 }}>
            <Alert severity="error">{error || 'Agent pattern catalog is unavailable.'}</Alert>
          </Box>
        ) : (
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', lg: '280px minmax(0, 1fr) 380px' },
              gap: 0,
              minHeight: 0,
            }}
          >
            <Box sx={{ minHeight: 0, overflow: 'auto', borderRight: { lg: 1 }, borderColor: 'divider', p: 1.5 }}>
              <BuilderNodePalette catalog={catalog} state={builderState} onAddNodeType={handleAddNodeType} />
            </Box>
            <Box sx={{ minHeight: 0, overflow: 'auto', p: 1.5 }}>
              {validationIssues.length > 0 ? (
                <Alert severity={validation?.valid ? 'success' : 'error'} sx={{ mb: 1 }}>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.25 }}>
                    {validationIssues.map((issue, index) => (
                      <Typography key={`${issue}-${index}`} variant="caption">
                        {issue}
                      </Typography>
                    ))}
                  </Box>
                </Alert>
              ) : null}
              <BuilderGraphEditor
                catalog={catalog}
                state={builderState}
                selection={selection}
                onSelectionChange={setSelection}
                onAddEdge={handleAddEdge}
              />
            </Box>
            <Box sx={{ minHeight: 0, overflow: 'auto', borderLeft: { lg: 1 }, borderColor: 'divider', p: 1.5 }}>
              <BuilderInspector
                catalog={catalog}
                state={builderState}
                selection={selection}
                onUpdateNode={handleUpdateNode}
                onUpdateEdge={handleUpdateEdge}
                onRemoveNode={handleRemoveNode}
                onRemoveEdge={handleRemoveEdge}
                onAddHitlGate={handleAddHitlGate}
              />
              <Divider sx={{ my: 1.5 }} />
              <Typography variant="caption" color="text.secondary">
                Nodes: {builderState.nodes.length} · Edges: {builderState.edges.length} · Tools: {builderState.allowed_tool_ids.length}
              </Typography>
            </Box>
          </Box>
        )}
      </Box>
    </ThemeProvider>
  );
}

