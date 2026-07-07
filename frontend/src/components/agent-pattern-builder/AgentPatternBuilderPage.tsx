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
  createInternalAgentPattern,
  getInternalAgentPatternCatalog,
  previewThreadAgentConfig,
  validateAgentPatternSpec,
  type AgentPatternCatalogResponse,
  type AgentPatternValidationReport,
  type ThreadAgentConfigPreviewResponse,
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
import BuilderPersistencePanel, {
  type BuilderBoundaryMessage,
  type BuilderPersistedPattern,
  type BuilderPersistenceState,
} from './BuilderPersistencePanel';
import BuilderValidationPanel from './BuilderValidationPanel';
import type { BuilderSelection, BuilderValidationIssue } from './types';

interface BuilderInternalBoundary {
  hasMetadata: boolean;
  authoringEnabled: boolean;
  runtimeEnabled: boolean;
  messages: BuilderBoundaryMessage[];
}

const collectNodeToolIds = (nodes: BuilderNodeState[]) => (
  Array.from(new Set(nodes.flatMap((node) => node.tool_contract_ids || []))).sort()
);

const slugifyTemplateId = (name: string) => {
  const slug = name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 72);
  return slug ? `internal_${slug}` : 'internal_custom_agent';
};

const edgeIndexFromSource = (state: AgentPatternBuilderState, sourceId: string) => (
  state.edges.findIndex((edge) => edge.from === sourceId)
);

const edgeIndexFromSourceTarget = (state: AgentPatternBuilderState, sourceId: string, targetId: string) => (
  state.edges.findIndex((edge) => edge.from === sourceId && (edge.to === targetId || Object.values(edge.routes || {}).includes(targetId)))
);

const inferIssueSelection = (state: AgentPatternBuilderState, message: string): BuilderSelection => {
  const explicitNode = message.match(/graph node ([^. ]+)/)?.[1]
    || message.match(/duplicate graph node id: ([^ ]+)/)?.[1]
    || message.match(/node_visit_limits\.([^ ]+)/)?.[1];
  if (explicitNode && state.nodes.some((node) => node.id === explicitNode)) {
    return { kind: 'node', nodeId: explicitNode };
  }

  const conditionalSource = message.match(/graph conditional edge from ([^ ]+)/)?.[1];
  if (conditionalSource) {
    const edgeIndex = edgeIndexFromSource(state, conditionalSource);
    if (edgeIndex >= 0) return { kind: 'edge', edgeIndex };
  }

  const incompatibleEdge = message.match(/node ([^ ]+) type [^ ]+ cannot connect to ([^ ]+) type/) || [];
  if (incompatibleEdge[1] && incompatibleEdge[2]) {
    const edgeIndex = edgeIndexFromSourceTarget(state, incompatibleEdge[1], incompatibleEdge[2]);
    if (edgeIndex >= 0) return { kind: 'edge', edgeIndex };
  }

  return null;
};

const buildValidationIssues = (
  state: AgentPatternBuilderState | null,
  validation: AgentPatternValidationReport | null,
): BuilderValidationIssue[] => {
  if (!state || !validation) return [];
  return [
    ...(validation.errors || []).map((message, index) => ({
      id: `error-${index}-${message}`,
      severity: 'error' as const,
      message,
      selection: inferIssueSelection(state, message),
    })),
    ...(validation.warnings || []).map((message, index) => ({
      id: `warning-${index}-${message}`,
      severity: 'warning' as const,
      message,
      selection: inferIssueSelection(state, message),
    })),
  ];
};

const hasExplicitFalse = (boundary: Record<string, any>, keys: string[]) => (
  keys.some((key) => boundary[key] === false)
);

const deriveInternalBoundary = (catalog: AgentPatternCatalogResponse | null): BuilderInternalBoundary => {
  const rawBoundary = catalog?.auth_boundary || {};
  const boundary = rawBoundary as Record<string, any>;
  const hasMetadata = Object.keys(boundary).length > 0;
  const authoringEnabled = !hasExplicitFalse(boundary, [
    'authoring_enabled',
    'internal_authoring_enabled',
    'internal_agent_pattern_authoring_enabled',
  ]);
  const runtimeEnabled = !hasExplicitFalse(boundary, [
    'custom_runtime_enabled',
    'custom_patterns_enabled',
    'runtime_custom_execution_enabled',
  ]);
  const messages: BuilderBoundaryMessage[] = [];

  if (!hasMetadata) {
    messages.push({
      severity: 'info',
      message: 'Catalog flag metadata is unavailable; authoring will rely on endpoint responses.',
    });
  }
  if (!authoringEnabled) {
    messages.push({
      severity: 'error',
      message: 'Internal pattern authoring is disabled by backend feature flags. Builder edits and saves are read-only.',
    });
  }
  if (!runtimeEnabled) {
    messages.push({
      severity: 'warning',
      message: 'Custom pattern runtime execution is disabled, so saved custom patterns are visible but will not run in chat yet.',
    });
  }

  return {
    hasMetadata,
    authoringEnabled,
    runtimeEnabled,
    messages,
  };
};

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
  const [threadPreviewId, setThreadPreviewId] = useState('');
  const [threadPreview, setThreadPreview] = useState<ThreadAgentConfigPreviewResponse | null>(null);
  const [threadPreviewError, setThreadPreviewError] = useState<string | null>(null);
  const [previewingThread, setPreviewingThread] = useState(false);
  const [persistenceForm, setPersistenceForm] = useState<BuilderPersistenceState>({
    templateId: 'internal_custom_agent',
    name: 'Internal Custom Agent',
    description: '',
    ownerId: '',
    changelog: '',
  });
  const [persistedPattern, setPersistedPattern] = useState<BuilderPersistedPattern | null>(null);
  const [busyAction, setBusyAction] = useState<'save' | null>(null);
  const [persistenceStatus, setPersistenceStatus] = useState<string | null>(null);
  const [persistenceError, setPersistenceError] = useState<string | null>(null);
  const boundary = useMemo(() => deriveInternalBoundary(catalog), [catalog]);
  const authoringDisabled = !boundary.authoringEnabled;

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
    if (!catalog || authoringDisabled) return;
    setBuilderState((previous) => {
      if (!previous) return previous;
      const next = normalizeBuilderState(catalog, updater(previous));
      return {
        ...next,
        allowed_tool_ids: collectNodeToolIds(next.nodes),
      };
    });
    setValidation(null);
    setThreadPreview(null);
    setThreadPreviewError(null);
    setPersistenceStatus(null);
  }, [authoringDisabled, catalog]);

  const resetToStarter = useCallback((nextStarter = starter) => {
    if (!catalog || authoringDisabled) return;
    setBuilderState(createInitialBuilderState(catalog, nextStarter));
    setSelection(null);
    setValidation(null);
    setThreadPreview(null);
    setThreadPreviewError(null);
    setPersistedPattern(null);
    setPersistenceStatus(null);
    setPersistenceError(null);
  }, [authoringDisabled, catalog, starter]);

  const handleStarterChange = (nextStarter: AgentPatternStarter) => {
    if (authoringDisabled) return;
    setStarter(nextStarter);
    resetToStarter(nextStarter);
  };

  const handleAddNodeType = (nodeType: string) => {
    if (!catalog || !builderState || authoringDisabled) return;
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
    if (!catalog || !builderState || authoringDisabled || !edge.to || !canConnectNodes(catalog, builderState, edge.from, edge.to).ok) return;
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
    if (!catalog || authoringDisabled) return;
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

  const spec = useMemo(() => (
    builderState ? assembleAgentPatternSpec(builderState) : null
  ), [builderState]);
  const validationIssues = useMemo(() => (
    buildValidationIssues(builderState, validation)
  ), [builderState, validation]);

  const handleThreadPreview = async () => {
    if (!threadPreviewId.trim()) return;
    try {
      setPreviewingThread(true);
      setThreadPreviewError(null);
      setThreadPreview(await previewThreadAgentConfig(threadPreviewId.trim()));
    } catch (err) {
      setThreadPreview(null);
      setThreadPreviewError(err instanceof Error ? err.message : String(err));
    } finally {
      setPreviewingThread(false);
    }
  };

  const updatePersistenceForm = (patch: Partial<BuilderPersistenceState>) => {
    setPersistenceForm((previous) => ({ ...previous, ...patch }));
    setPersistenceStatus(null);
    setPersistenceError(null);
  };

  const handleGenerateTemplateId = () => {
    if (authoringDisabled) return;
    updatePersistenceForm({ templateId: slugifyTemplateId(persistenceForm.name) });
  };

  const handleSaveInternalVersion = async () => {
    if (!builderState || !spec) return;
    if (authoringDisabled) {
      setPersistenceError('Internal pattern authoring is disabled by backend feature flags.');
      return;
    }
    try {
      setBusyAction('save');
      setPersistenceError(null);
      setPersistenceStatus(null);
      const templateId = persistenceForm.templateId.trim();
      const saveSpec = {
        ...spec,
        pattern_type: templateId || spec.pattern_type,
      };
      const report = await validateAgentPatternSpec(saveSpec);
      setValidation(report);
      if (!report.valid) {
        setPersistenceError('Validation failed. Fix the reported issues before saving.');
        return;
      }
      const response = await createInternalAgentPattern({
        template_id: templateId,
        name: persistenceForm.name.trim(),
        description: persistenceForm.description,
        owner_id: persistenceForm.ownerId.trim() || null,
        changelog: persistenceForm.changelog || null,
        spec_json: saveSpec,
        set_current: true,
      });
      setPersistedPattern({ template: response.agent_pattern, version: response.version });
      setPersistenceStatus(`Saved ${response.agent_pattern.id} v${response.version.version}.`);
    } catch (err) {
      setPersistenceError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyAction(null);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', display: 'grid', gridTemplateRows: 'auto minmax(0, 1fr)', bgcolor: 'background.default' }}>
        <BuilderActionsBar
          starter={starter}
          disabled={authoringDisabled}
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
              <BuilderNodePalette
                catalog={catalog}
                state={builderState}
                disabled={authoringDisabled}
                onAddNodeType={handleAddNodeType}
              />
            </Box>
            <Box sx={{ minHeight: 0, overflow: 'auto', p: 1.5 }}>
              {spec ? (
                <BuilderValidationPanel
                  catalog={catalog}
                  spec={spec}
                  validation={validation}
                  issues={validationIssues}
                  threadPreviewId={threadPreviewId}
                  onThreadPreviewIdChange={setThreadPreviewId}
                  onPreviewThread={handleThreadPreview}
                  previewing={previewingThread}
                  previewResult={threadPreview}
                  previewError={threadPreviewError}
                  onSelectIssue={setSelection}
                />
              ) : null}
              <BuilderGraphEditor
                catalog={catalog}
                state={builderState}
                selection={selection}
                validationIssues={validationIssues}
                disabled={authoringDisabled}
                onSelectionChange={setSelection}
                onAddEdge={handleAddEdge}
              />
            </Box>
            <Box sx={{ minHeight: 0, overflow: 'auto', borderLeft: { lg: 1 }, borderColor: 'divider', p: 1.5 }}>
              <BuilderPersistencePanel
                form={persistenceForm}
                onFormChange={updatePersistenceForm}
                persisted={persistedPattern}
                busyAction={busyAction}
                statusMessage={persistenceStatus}
                errorMessage={persistenceError}
                canSave={Boolean(spec && persistenceForm.templateId.trim() && persistenceForm.name.trim())}
                authoringDisabled={authoringDisabled}
                boundaryMessages={boundary.messages}
                onGenerateTemplateId={handleGenerateTemplateId}
                onSave={handleSaveInternalVersion}
              />
              <Divider sx={{ my: 1.5 }} />
              <BuilderInspector
                catalog={catalog}
                state={builderState}
                selection={selection}
                disabled={authoringDisabled}
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
