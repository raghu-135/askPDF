import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  Button,
  CircularProgress,
  CssBaseline,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  List,
  ListItemButton,
  ListItemText,
  Tab,
  Tabs,
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../../theme';
import {
  saveInternalAgentWorkflow,
  deleteInternalAgentWorkflow,
  getInternalAgentWorkflow,
  getInternalAgentWorkflowCatalog,
  listAgentWorkflows,
  previewThreadAgentConfig,
  validateAgentWorkflowSpec,
  type AgentWorkflow,
  type AgentWorkflowCatalogResponse,
  type AgentWorkflowValidationReport,
  type ThreadAgentConfigPreviewResponse,
} from '../../lib/api';
import {
  assembleAgentWorkflowSpec,
  canAddNodeType,
  canConnectNodes,
  canInsertExistingNodeBefore,
  canInsertNodeTypeBefore,
  createHitlGateForTarget,
  createInitialBuilderState,
  getIncomingPaths,
  getAllowedToolContractsForNode,
  getCanonicalNodeId,
  getAllowedRouteFunctionsForNode,
  getRouteLabelsForFunction,
  insertNodeBefore,
  loadBuilderStateFromSpec,
  normalizeBuilderState,
  type AgentWorkflowBuilderState,
  type AgentWorkflowStarter,
  type BuilderEdgeState,
  type BuilderIncomingPath,
  type BuilderNodeState,
} from '../../lib/agent-workflow-builder';
import BuilderActionsBar from './BuilderActionsBar';
import BuilderGraphEditor from './BuilderGraphEditor';
import BuilderInspector from './BuilderInspector';
import BuilderNodePalette from './BuilderNodePalette';
import BuilderPersistencePanel, {
  type BuilderBoundaryMessage,
  type BuilderPersistedWorkflow,
  type BuilderPersistenceState,
} from './BuilderPersistencePanel';
import BuilderValidationPanel from './BuilderValidationPanel';
import BuilderTestStudio from './BuilderTestStudio';
import BuilderResizeHandle from './BuilderResizeHandle';
import type { BuilderSelection, BuilderValidationIssue } from './types';

interface BuilderInternalBoundary {
  hasMetadata: boolean;
  authoringEnabled: boolean;
  runtimeEnabled: boolean;
  messages: BuilderBoundaryMessage[];
}

type ContextualNodeRequest =
  | { mode: 'after'; source: string; route?: string }
  | { mode: 'before'; target: string; incomingPaths: BuilderIncomingPath[]; selectedPathId?: string };

const BUILDER_LAYOUT_STORAGE_KEY = 'askpdf.agentWorkflowBuilder.layout.v1';
const DEFAULT_BUILDER_LAYOUT = {
  sidebarWidth: 360,
  palettePercent: 40,
  graphElementsHeight: 104,
  graphElementsCollapsed: false,
};

const readBuilderLayout = () => {
  if (typeof window === 'undefined') return DEFAULT_BUILDER_LAYOUT;
  try {
    const stored = JSON.parse(window.localStorage.getItem(BUILDER_LAYOUT_STORAGE_KEY) || '{}');
    return {
      sidebarWidth: Math.min(560, Math.max(300, Number(stored.sidebarWidth) || 360)),
      palettePercent: Math.min(65, Math.max(25, Number(stored.palettePercent) || 40)),
      graphElementsHeight: Math.min(320, Math.max(72, Number(stored.graphElementsHeight) || 104)),
      graphElementsCollapsed: Boolean(stored.graphElementsCollapsed),
    };
  } catch {
    return DEFAULT_BUILDER_LAYOUT;
  }
};

const collectNodeToolIds = (nodes: BuilderNodeState[]) => (
  Array.from(new Set(nodes.flatMap((node) => node.tool_contract_ids || []))).sort()
);

const BUILTIN_STARTERS: AgentWorkflowStarter[] = ['router', 'plan_execute', 'evaluator_replanner'];

const isBuiltinStarter = (value: string): value is AgentWorkflowStarter => (
  BUILTIN_STARTERS.includes(value as AgentWorkflowStarter)
);

const customStarterValue = (workflowId: string) => `custom:${workflowId}`;

const workflowIdFromCustomStarter = (value: string) => (
  value.startsWith('custom:') ? value.slice('custom:'.length) : null
);

const edgeIndexFromSource = (state: AgentWorkflowBuilderState, sourceId: string) => (
  state.edges.findIndex((edge) => edge.from === sourceId)
);

const edgeIndexFromSourceTarget = (state: AgentWorkflowBuilderState, sourceId: string, targetId: string) => (
  state.edges.findIndex((edge) => edge.from === sourceId && (edge.to === targetId || Object.values(edge.routes || {}).includes(targetId)))
);

const inferIssueSelection = (state: AgentWorkflowBuilderState, message: string): BuilderSelection => {
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
  state: AgentWorkflowBuilderState | null,
  validation: AgentWorkflowValidationReport | null,
): BuilderValidationIssue[] => {
  if (!state) return [];
  const local: BuilderValidationIssue[] = [];
  if (!state.edges.some((edge) => edge.from === 'START')) {
    local.push({ id: 'local-missing-start', code: 'missing_start', severity: 'error', message: 'Connect Start to the first workflow step.', selection: null, fix: { kind: 'missing_start' } });
  }
  if (!state.edges.some((edge) => edge.to === 'END' || Object.values(edge.routes || {}).includes('END'))) {
    local.push({ id: 'local-missing-end', code: 'missing_end', severity: 'error', message: 'Connect the final step to End.', selection: null, fix: { kind: 'missing_end' } });
  }
  if (!validation) return local;
  if (validation.issues?.length) {
    const backend = validation.issues.map((issue, index): BuilderValidationIssue => ({
      id: `${issue.code}-${index}-${issue.message}`,
      code: issue.code,
      severity: issue.severity,
      message: issue.message,
      selection: typeof issue.edge_index === 'number'
        ? { kind: 'edge', edgeIndex: issue.edge_index }
        : issue.node_id && state.nodes.some((node) => node.id === issue.node_id)
          ? { kind: 'node', nodeId: issue.node_id }
          : inferIssueSelection(state, issue.message),
      fix: issue.fix,
    }));
    return [...local.filter((item) => !backend.some((issue) => issue.code === item.code)), ...backend];
  }
  return [
    ...local,
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

const deriveInternalBoundary = (catalog: AgentWorkflowCatalogResponse | null): BuilderInternalBoundary => {
  const rawBoundary = catalog?.auth_boundary || {};
  const boundary = rawBoundary as Record<string, any>;
  const hasMetadata = Object.keys(boundary).length > 0;
  const authoringEnabled = !hasExplicitFalse(boundary, [
    'authoring_enabled',
    'internal_authoring_enabled',
    'internal_agent_workflow_authoring_enabled',
  ]);
  const runtimeEnabled = !hasExplicitFalse(boundary, [
    'custom_runtime_enabled',
    'custom_workflows_enabled',
    'runtime_custom_execution_enabled',
  ]);
  const messages: BuilderBoundaryMessage[] = [];

  if (!hasMetadata) {
    return {
      hasMetadata,
      authoringEnabled,
      runtimeEnabled,
      messages,
    };
  }
  if (!authoringEnabled) {
    messages.push({
      severity: 'error',
      message: 'Internal workflow authoring is disabled by backend feature flags. Builder edits and saves are read-only.',
    });
  }
  if (!runtimeEnabled) {
    messages.push({
      severity: 'warning',
      message: 'Custom workflow runtime execution is disabled, so saved custom workflows are visible but will not run in chat yet.',
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

export default function AgentWorkflowBuilderPage() {
  const darkMode = usePrefersDarkMode();
  const theme = useMemo(() => getTheme(darkMode), [darkMode]);
  const [catalog, setCatalog] = useState<AgentWorkflowCatalogResponse | null>(null);
  const [customWorkflows, setCustomWorkflows] = useState<AgentWorkflow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [starter, setStarter] = useState<string>('router');
  const [builderState, setBuilderState] = useState<AgentWorkflowBuilderState | null>(null);
  const [selection, setSelection] = useState<BuilderSelection>(null);
  const [validation, setValidation] = useState<AgentWorkflowValidationReport | null>(null);
  const [validating, setValidating] = useState(false);
  const [threadPreviewId, setThreadPreviewId] = useState('');
  const [threadPreview, setThreadPreview] = useState<ThreadAgentConfigPreviewResponse | null>(null);
  const [threadPreviewError, setThreadPreviewError] = useState<string | null>(null);
  const [previewingThread, setPreviewingThread] = useState(false);
  const [persistenceForm, setPersistenceForm] = useState<BuilderPersistenceState>({
    workflowId: '',
    name: 'Internal Custom Agent',
    description: '',
  });
  const [persistedWorkflow, setPersistedWorkflow] = useState<BuilderPersistedWorkflow | null>(null);
  const [busyAction, setBusyAction] = useState<'save' | 'delete' | null>(null);
  const [persistenceStatus, setPersistenceStatus] = useState<string | null>(null);
  const [persistenceError, setPersistenceError] = useState<string | null>(null);
  const boundary = useMemo(() => deriveInternalBoundary(catalog), [catalog]);
  const authoringDisabled = !boundary.authoringEnabled;
  const undoStack = useRef<AgentWorkflowBuilderState[]>([]);
  const redoStack = useRef<AgentWorkflowBuilderState[]>([]);
  const [historyVersion, setHistoryVersion] = useState(0);
  const [isDirty, setIsDirty] = useState(false);
  const [nodeRequest, setNodeRequest] = useState<ContextualNodeRequest | null>(null);
  const [workspace, setWorkspace] = useState<'build' | 'test'>('build');
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [builderLayout, setBuilderLayout] = useState(DEFAULT_BUILDER_LAYOUT);
  const utilityRailRef = useRef<HTMLDivElement | null>(null);
  const layoutHydrated = useRef(false);

  useEffect(() => {
    setBuilderLayout(readBuilderLayout());
  }, []);

  useEffect(() => {
    if (!layoutHydrated.current) {
      layoutHydrated.current = true;
      return;
    }
    if (typeof window === 'undefined') return;
    try {
      window.localStorage.setItem(BUILDER_LAYOUT_STORAGE_KEY, JSON.stringify(builderLayout));
    } catch {
      // Layout persistence is best-effort and must never block authoring.
    }
  }, [builderLayout]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([
      getInternalAgentWorkflowCatalog(),
      listAgentWorkflows().catch(() => ({ agent_workflows: [] })),
    ])
      .then(([nextCatalog, patternList]) => {
        if (cancelled) return;
        setCatalog(nextCatalog);
        setCustomWorkflows((patternList.agent_workflows || []).filter((pattern) => !pattern.is_builtin));
        setBuilderState(createInitialBuilderState(nextCatalog, 'router'));
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

  const refreshCustomWorkflows = useCallback(async () => {
    const response = await listAgentWorkflows();
    setCustomWorkflows((response.agent_workflows || []).filter((pattern) => !pattern.is_builtin));
  }, []);

  const updateState = useCallback((updater: (previous: AgentWorkflowBuilderState) => AgentWorkflowBuilderState) => {
    if (!catalog || authoringDisabled) return;
    setBuilderState((previous) => {
      if (!previous) return previous;
      undoStack.current.push(JSON.parse(JSON.stringify(previous)));
      if (undoStack.current.length > 50) undoStack.current.shift();
      redoStack.current = [];
      const next = normalizeBuilderState(catalog, updater(previous));
      return {
        ...next,
        allowed_tool_ids: collectNodeToolIds(next.nodes),
      };
    });
    setValidation(null);
    setHistoryVersion((value) => value + 1);
    setIsDirty(true);
    setThreadPreview(null);
    setThreadPreviewError(null);
    setPersistenceStatus(null);
  }, [authoringDisabled, catalog]);

  const handleUndo = useCallback(() => {
    if (!builderState || undoStack.current.length === 0) return;
    const previous = undoStack.current.pop()!;
    redoStack.current.push(JSON.parse(JSON.stringify(builderState)));
    setBuilderState(previous);
    setSelection(null);
    setValidation(null);
    setHistoryVersion((value) => value + 1);
    setIsDirty(true);
  }, [builderState]);

  const handleRedo = useCallback(() => {
    if (!builderState || redoStack.current.length === 0) return;
    const next = redoStack.current.pop()!;
    undoStack.current.push(JSON.parse(JSON.stringify(builderState)));
    setBuilderState(next);
    setSelection(null);
    setValidation(null);
    setHistoryVersion((value) => value + 1);
    setIsDirty(true);
  }, [builderState]);

  useEffect(() => {
    if (!builderState) return;
    const timer = window.setTimeout(() => {
      void validateAgentWorkflowSpec(assembleAgentWorkflowSpec(builderState))
        .then(setValidation)
        .catch(() => undefined);
    }, 500);
    return () => window.clearTimeout(timer);
  }, [builderState]);

  useEffect(() => {
    if (!isDirty) return;
    const handler = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [isDirty]);

  const resetToStarter = useCallback((nextStarter: AgentWorkflowStarter = 'router') => {
    if (!catalog || authoringDisabled) return;
    setBuilderState(createInitialBuilderState(catalog, nextStarter));
    setStarter(nextStarter);
    setSelection(null);
    setValidation(null);
    setThreadPreview(null);
    setThreadPreviewError(null);
    setPersistedWorkflow(null);
    setPersistenceStatus(null);
    setPersistenceError(null);
    undoStack.current = [];
    redoStack.current = [];
    setHistoryVersion((value) => value + 1);
    setIsDirty(false);
  }, [authoringDisabled, catalog]);

  const loadCustomWorkflow = useCallback(async (workflowId: string) => {
    if (!catalog || authoringDisabled) return;
    try {
      setError(null);
      const response = await getInternalAgentWorkflow(workflowId);
      const loadedState = normalizeBuilderState(catalog, loadBuilderStateFromSpec(response.spec.spec_json));
      setBuilderState({
        ...loadedState,
        allowed_tool_ids: collectNodeToolIds(loadedState.nodes),
      });
      setStarter(customStarterValue(response.agent_workflow.id));
      setPersistenceForm({
        workflowId: response.agent_workflow.id,
        name: response.agent_workflow.name || response.agent_workflow.id,
        description: response.agent_workflow.description || '',
      });
      setPersistedWorkflow({ workflow: response.agent_workflow, spec: response.spec });
      setSelection(null);
      setValidation(null);
      setThreadPreview(null);
      setThreadPreviewError(null);
      setPersistenceStatus(`Loaded ${response.agent_workflow.name || response.agent_workflow.id}.`);
      setPersistenceError(null);
      undoStack.current = [];
      redoStack.current = [];
      setHistoryVersion((value) => value + 1);
      setIsDirty(false);
    } catch (err) {
      setPersistenceError(err instanceof Error ? err.message : String(err));
    }
  }, [authoringDisabled, catalog]);

  const handleStarterChange = (nextStarter: AgentWorkflowStarter | string) => {
    if (authoringDisabled) return;
    const customWorkflowId = workflowIdFromCustomStarter(nextStarter);
    if (customWorkflowId) {
      void loadCustomWorkflow(customWorkflowId);
      return;
    }
    if (!isBuiltinStarter(nextStarter)) return;
    resetToStarter(nextStarter);
  };

  const handleAddNodeType = (nodeType: string, position?: { x: number; y: number }) => {
    if (!catalog || !builderState || authoringDisabled) return;
    const compatibility = canAddNodeType(catalog, builderState, nodeType);
    if (!compatibility.ok) return;
    const id = getCanonicalNodeId(nodeType, builderState.nodes.map((node) => node.id));
    const allowedTools = getAllowedToolContractsForNode(catalog, nodeType);
    const nextNode: BuilderNodeState = {
      id,
      type: nodeType,
      ...(position ? { position } : {}),
      ...(allowedTools[0] ? { tool_contract_ids: [allowedTools[0].id] } : {}),
    };
    updateState((previous) => ({
      ...previous,
      nodes: [...previous.nodes, nextNode],
    }));
    setSelection({ kind: 'node', nodeId: id });
  };

  const handleAddNote = () => {
    if (typeof window === 'undefined') return;
    const text = window.prompt('What should this canvas note say?')?.trim();
    if (!text) return;
    updateState((previous) => ({
      ...previous,
      builder_ui: {
        ...(previous.builder_ui || {}),
        notes: [
          ...(previous.builder_ui?.notes || []),
          { id: `note-${Date.now()}`, text, position: { x: 80, y: 80 } },
        ],
      },
    }));
  };
  const handleAddGroup = () => {
    if (typeof window === 'undefined' || !builderState) return;
    const label = window.prompt('Name this visual group:')?.trim();
    if (!label) return;
    updateState((previous) => ({
      ...previous,
      builder_ui: {
        ...(previous.builder_ui || {}),
        groups: [
          ...(previous.builder_ui?.groups || []),
          { id: `group-${Date.now()}`, label, node_ids: previous.nodes.map((node) => node.id), position: { x: 40, y: 40 } },
        ],
      },
    }));
  };
  const handleUpdateNote = (id: string, position: { x: number; y: number }) => updateState((previous) => ({
    ...previous,
    builder_ui: {
      ...(previous.builder_ui || {}),
      notes: (previous.builder_ui?.notes || []).map((note) => note.id === id ? { ...note, position } : note),
      groups: (previous.builder_ui?.groups || []).map((group) => group.id === id ? { ...group, position } : group),
    },
  }));
  const handleRemoveNote = (id: string) => updateState((previous) => ({
    ...previous,
    builder_ui: {
      ...(previous.builder_ui || {}),
      notes: (previous.builder_ui?.notes || []).filter((note) => note.id !== id),
      groups: (previous.builder_ui?.groups || []).filter((group) => group.id !== id),
    },
  }));
  const handlePositionsChange = (positions: Record<string, { x: number; y: number }>) => updateState((previous) => ({
    ...previous,
    nodes: previous.nodes.map((node) => positions[node.id] ? { ...node, position: positions[node.id] } : node),
    builder_ui: {
      ...(previous.builder_ui || {}),
      notes: (previous.builder_ui?.notes || []).map((note) => positions[note.id] ? { ...note, position: positions[note.id] } : note),
      groups: (previous.builder_ui?.groups || []).map((group) => positions[group.id] ? { ...group, position: positions[group.id] } : group),
    },
  }));

  const connectState = useCallback((
    previous: AgentWorkflowBuilderState,
    source: string,
    target: string,
    route?: string,
  ): AgentWorkflowBuilderState => {
    if (!catalog || !canConnectNodes(catalog, previous, source, target).ok) return previous;
    if (route) {
      const sourceType = previous.nodes.find((node) => node.id === source)?.type;
      const routeFn = sourceType ? getAllowedRouteFunctionsForNode(catalog, sourceType)[0] : undefined;
      if (!routeFn) return previous;
      const edgeIndex = previous.edges.findIndex((edge) => edge.from === source && edge.conditional);
      if (edgeIndex >= 0) {
        return {
          ...previous,
          edges: previous.edges.map((edge, index) => index === edgeIndex
            ? { ...edge, route_fn: routeFn, routes: { ...(edge.routes || {}), [route]: target } }
            : edge),
        };
      }
      return {
        ...previous,
        edges: [...previous.edges, { from: source, conditional: true, route_fn: routeFn, routes: { [route]: target } }],
      };
    }
    const duplicate = previous.edges.some((edge) => !edge.conditional && edge.from === source && edge.to === target);
    return duplicate ? previous : { ...previous, edges: [...previous.edges, { from: source, to: target }] };
  }, [catalog]);

  const handleConnectNodes = useCallback((source: string, target: string, route?: string) => {
    updateState((previous) => connectState(previous, source, target, route));
  }, [connectState, updateState]);

  const handleRequestAddNext = useCallback((source: string, route?: string) => {
    if (!catalog || !builderState) return;
    const sourceType = builderState.nodes.find((node) => node.id === source)?.type;
    const routeFn = sourceType ? getAllowedRouteFunctionsForNode(catalog, sourceType)[0] : undefined;
    const inferredRoute = route || (routeFn ? getRouteLabelsForFunction(catalog, routeFn)?.[0] : undefined);
    setNodeRequest({ mode: 'after', source, route: inferredRoute });
  }, [builderState, catalog]);

  const handleRequestAddPrevious = useCallback((target: string) => {
    if (!builderState) return;
    const incomingPaths = getIncomingPaths(builderState, target);
    setNodeRequest({
      mode: 'before',
      target,
      incomingPaths,
      ...(incomingPaths.length === 1 ? { selectedPathId: incomingPaths[0].id } : {}),
    });
  }, [builderState]);

  const positionForPreviousStep = useCallback((
    state: AgentWorkflowBuilderState,
    target: string,
    incomingPath?: BuilderIncomingPath,
  ) => {
    const targetPosition = state.nodes.find((node) => node.id === target)?.position;
    const sourcePosition = incomingPath
      ? state.nodes.find((node) => node.id === incomingPath.source)?.position
      : undefined;
    if (sourcePosition && targetPosition) {
      return {
        x: (sourcePosition.x + targetPosition.x) / 2,
        y: (sourcePosition.y + targetPosition.y) / 2,
      };
    }
    if (targetPosition) return { x: targetPosition.x - 320, y: targetPosition.y };
    if (sourcePosition) return { x: sourcePosition.x + 320, y: sourcePosition.y };
    return { x: 0, y: 0 };
  }, []);

  const handleUpdateNode = (nodeId: string, patch: Partial<BuilderNodeState>) => {
    updateState((previous) => ({
      ...previous,
      nodes: previous.nodes.map((node) => (node.id === nodeId ? { ...node, ...patch } : node)),
    }));
  };

  const handleNodePositionChange = (nodeId: string, position: { x: number; y: number }) => {
    handleUpdateNode(nodeId, { position });
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

  const handleUpdateEdge = (edgeIndex: number, patch: Partial<BuilderEdgeState>) => {
    updateState((previous) => ({
      ...previous,
      edges: previous.edges.map((edge, index) => (index === edgeIndex ? { ...edge, ...patch } : edge)),
    }));
  };

  const handleRemoveEdge = (edgeIndex: number, route?: string) => {
    updateState((previous) => ({
      ...previous,
      edges: previous.edges.flatMap((edge, index) => {
        if (index !== edgeIndex) return [edge];
        if (!route || !edge.conditional) return [];
        const routes = Object.fromEntries(Object.entries(edge.routes || {}).filter(([label]) => label !== route));
        return Object.keys(routes).length ? [{ ...edge, routes }] : [];
      }),
    }));
    setSelection(null);
  };

  const handleApplyFix = (issue: BuilderValidationIssue) => {
    if (!catalog || !builderState || !issue.fix) return;
    if (issue.fix.requires_confirmation && typeof window !== 'undefined' && !window.confirm(`Apply this suggested fix?\n\n${issue.message}`)) return;
    if (issue.fix.kind === 'missing_start') {
      const target = builderState.nodes.find((node) => canConnectNodes(catalog, builderState, 'START', node.id).ok);
      if (target) handleConnectNodes('START', target.id);
      return;
    }
    if (issue.fix.kind === 'missing_end') {
      const candidates = [...builderState.nodes].reverse();
      const source = candidates.find((node) => canConnectNodes(catalog, builderState, node.id, 'END').ok);
      if (source) handleConnectNodes(source.id, 'END');
      return;
    }
    if ((issue.fix.kind === 'incompatible_connection' || issue.fix.kind === 'invalid_edge') && issue.selection?.kind === 'edge') {
      handleRemoveEdge(issue.selection.edgeIndex);
      return;
    }
    if (issue.fix.kind === 'unreachable_node' && issue.fix.node_id) {
      const target = String(issue.fix.node_id);
      const sources = ['START', ...builderState.nodes.map((node) => node.id)];
      const source = sources.find((id) => id !== target && canConnectNodes(catalog, builderState, id, target).ok);
      if (source) handleConnectNodes(source, target);
      return;
    }
    if (issue.fix.kind === 'missing_route' && issue.fix.node_id) {
      const source = String(issue.fix.node_id);
      const route = issue.message.match(/route[s]? ['\"]?([a-z0-9_]+)['\"]?/i)?.[1];
      const target = [...builderState.nodes.map((node) => node.id), 'END']
        .find((id) => id !== source && canConnectNodes(catalog, builderState, source, id).ok);
      if (route && target) handleConnectNodes(source, target, route);
    }
  };

  const handleAddHitlGate = (targetNodeId: string) => {
    if (!catalog || authoringDisabled) return;
    updateState((previous) => createHitlGateForTarget(catalog, previous, targetNodeId));
  };

  const handleValidate = async () => {
    if (!builderState) return;
    try {
      setValidating(true);
      const report = await validateAgentWorkflowSpec(assembleAgentWorkflowSpec(builderState));
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
    builderState ? assembleAgentWorkflowSpec(builderState) : null
  ), [builderState]);
  const validationIssues = useMemo(() => (
    buildValidationIssues(builderState, validation)
  ), [builderState, validation]);
  const baseWorkflowId = workflowIdFromCustomStarter(starter) || ({
    router: 'router_rag_agent',
    plan_execute: 'plan_execute_rag_agent',
    evaluator_replanner: 'evaluator_replanner_rag_agent',
  } as Record<string, string>)[starter] || String(spec?.workflow_id || 'router_rag_agent');

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

  const handleSaveInternalWorkflow = async () => {
    if (!builderState || !spec) return;
    if (authoringDisabled) {
      setPersistenceError('Internal workflow authoring is disabled by backend feature flags.');
      return;
    }
    try {
      setBusyAction('save');
      setPersistenceError(null);
      setPersistenceStatus(null);
      const workflowId = persistedWorkflow?.workflow.id;
      const saveSpec = { ...spec };
      const report = await validateAgentWorkflowSpec(saveSpec);
      setValidation(report);
      if (!report.valid) {
        setPersistenceError('Validation failed. Fix the reported issues before saving.');
        return;
      }
      const response = await saveInternalAgentWorkflow({
        ...(workflowId ? { workflow_id: workflowId } : {}),
        name: persistenceForm.name.trim(),
        description: persistenceForm.description,
        spec_json: saveSpec,
      });
      setPersistedWorkflow({ workflow: response.agent_workflow, spec: response.spec });
      setPersistenceForm((previous) => ({
        ...previous,
        workflowId: response.agent_workflow.id,
      }));
      setStarter(customStarterValue(response.agent_workflow.id));
      await refreshCustomWorkflows();
      setPersistenceStatus(`Saved ${response.agent_workflow.name || response.agent_workflow.id}.`);
      setIsDirty(false);
      setSaveDialogOpen(false);
    } catch (err) {
      setPersistenceError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyAction(null);
    }
  };

  const handleDeleteInternalWorkflow = async () => {
    if (!persistedWorkflow || persistedWorkflow.workflow.is_builtin || authoringDisabled) return;
    const workflowId = persistedWorkflow.workflow.id;
    if (typeof window !== 'undefined' && !window.confirm(`Delete custom agent workflow "${workflowId}"?`)) {
      return;
    }
    try {
      setBusyAction('delete');
      setPersistenceError(null);
      setPersistenceStatus(null);
      await deleteInternalAgentWorkflow(workflowId);
      await refreshCustomWorkflows();
      setPersistenceForm((previous) => ({
        ...previous,
        workflowId: '',
      }));
      resetToStarter('router');
      setPersistedWorkflow(null);
      setPersistenceStatus(`Deleted ${workflowId}.`);
      setSaveDialogOpen(false);
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
          customWorkflows={customWorkflows}
          disabled={authoringDisabled}
          onStarterChange={handleStarterChange}
          onReset={() => resetToStarter()}
          onValidate={handleValidate}
          validating={validating}
          validation={validation}
          dirty={isDirty}
          canUndo={undoStack.current.length > 0}
          canRedo={redoStack.current.length > 0}
          onUndo={handleUndo}
          onRedo={handleRedo}
          onOpenSave={() => setSaveDialogOpen(true)}
          saveBusy={busyAction === 'save'}
          savedWorkflowId={persistedWorkflow?.workflow.id}
          workflowName={persistenceForm.name}
        />
        {loading ? (
          <Box sx={{ display: 'grid', placeItems: 'center' }}>
            <CircularProgress />
          </Box>
        ) : error || !catalog || !builderState ? (
          <Box sx={{ p: 2 }}>
            <Alert severity="error">{error || 'Agent workflow catalog is unavailable.'}</Alert>
          </Box>
        ) : (
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: workspace === 'test'
                ? 'minmax(0, 1fr)'
                : { xs: '1fr', lg: `minmax(0, 1fr) 8px ${builderLayout.sidebarWidth}px` },
              gap: 0,
              minHeight: 0,
              overflow: { xs: 'auto', lg: 'hidden' },
            }}
          >
            <Box sx={{ minHeight: 0, overflow: 'auto', p: 1.5, display: 'flex', flexDirection: 'column' }}>
              <Tabs value={workspace} onChange={(_, value) => setWorkspace(value)} sx={{ mb: 1 }}>
                <Tab value="build" label="Build" />
                <Tab value="test" label="Test" />
              </Tabs>
              {workspace === 'build' && spec ? (
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
                  onApplyFix={handleApplyFix}
                />
              ) : null}
              {workspace === 'build' ? (
                <BuilderGraphEditor
                  catalog={catalog}
                  state={builderState}
                  selection={selection}
                  validationIssues={validationIssues}
                  disabled={authoringDisabled}
                  onSelectionChange={setSelection}
                  onConnectNodes={handleConnectNodes}
                  onRemoveEdge={handleRemoveEdge}
                  onRemoveNode={handleRemoveNode}
                  onNodePositionChange={handleNodePositionChange}
                  onAddNodeAt={handleAddNodeType}
                  onRequestAddPrevious={handleRequestAddPrevious}
                  onRequestAddNext={handleRequestAddNext}
                  onUpdateNote={handleUpdateNote}
                  onRemoveNote={handleRemoveNote}
                  onPositionsChange={handlePositionsChange}
                  graphElementsHeight={builderLayout.graphElementsHeight}
                  graphElementsCollapsed={builderLayout.graphElementsCollapsed}
                  onGraphElementsHeightChange={(graphElementsHeight) => setBuilderLayout((previous) => ({ ...previous, graphElementsHeight }))}
                  onGraphElementsCollapsedChange={(graphElementsCollapsed) => setBuilderLayout((previous) => ({ ...previous, graphElementsCollapsed }))}
                />
              ) : spec ? (
                <BuilderTestStudio spec={spec} baseWorkflowId={baseWorkflowId} />
              ) : null}
            </Box>
            {workspace === 'build' ? (
              <BuilderResizeHandle
                orientation="vertical"
                value={builderLayout.sidebarWidth}
                min={300}
                max={560}
                defaultValue={360}
                direction={-1}
                label="Resize workflow utility sidebar"
                onChange={(sidebarWidth) => setBuilderLayout((previous) => ({ ...previous, sidebarWidth }))}
                sx={{ display: { xs: 'none', lg: 'block' } }}
              />
            ) : null}
            {workspace === 'build' ? (
              <Box ref={utilityRailRef} sx={{ minHeight: 0, overflow: { xs: 'auto', lg: 'hidden' }, borderLeft: { lg: 1 }, borderColor: 'divider' }}>
                <Box
                  sx={{
                    display: { xs: 'none', lg: 'grid' },
                    height: '100%',
                    gridTemplateRows: `minmax(180px, ${builderLayout.palettePercent}fr) 8px minmax(240px, ${100 - builderLayout.palettePercent}fr)`,
                    minHeight: 0,
                  }}
                >
                  <Box sx={{ minHeight: 0, overflow: 'auto', p: 1.5 }}>
                    <BuilderNodePalette catalog={catalog} state={builderState} disabled={authoringDisabled} onAddNodeType={handleAddNodeType} onAddNote={handleAddNote} onAddGroup={handleAddGroup} />
                  </Box>
                  <BuilderResizeHandle
                    orientation="horizontal"
                    value={builderLayout.palettePercent}
                    min={25}
                    max={65}
                    defaultValue={40}
                    step={2}
                    label="Resize Node Palette and Inspector"
                    getDragScale={() => 100 / Math.max(1, utilityRailRef.current?.clientHeight || 700)}
                    onChange={(palettePercent) => setBuilderLayout((previous) => ({ ...previous, palettePercent }))}
                  />
                  <Box sx={{ minHeight: 0, overflow: 'auto', p: 1.5 }}>
                    <BuilderInspector
                      catalog={catalog} state={builderState} selection={selection} disabled={authoringDisabled}
                      onUpdateNode={handleUpdateNode} onUpdateEdge={handleUpdateEdge} onRemoveNode={handleRemoveNode}
                      onRemoveEdge={handleRemoveEdge} onAddHitlGate={handleAddHitlGate}
                      onUpdateSettings={(patch) => updateState((previous) => ({ ...previous, extraConfig: { ...(previous.extraConfig || {}), ...patch } }))}
                    />
                    <Divider sx={{ my: 1.5 }} />
                    <Typography variant="caption" color="text.secondary">Nodes: {builderState.nodes.length} · Edges: {builderState.edges.length} · Tools: {builderState.allowed_tool_ids.length}</Typography>
                  </Box>
                </Box>
                <Box sx={{ display: { xs: 'block', lg: 'none' }, p: 1 }}>
                  <Accordion defaultExpanded disableGutters>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}><Typography variant="subtitle2">Node Palette</Typography></AccordionSummary>
                    <AccordionDetails><BuilderNodePalette catalog={catalog} state={builderState} disabled={authoringDisabled} onAddNodeType={handleAddNodeType} onAddNote={handleAddNote} onAddGroup={handleAddGroup} /></AccordionDetails>
                  </Accordion>
                  <Accordion defaultExpanded disableGutters>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}><Typography variant="subtitle2">Inspector</Typography></AccordionSummary>
                    <AccordionDetails>
                      <BuilderInspector
                        catalog={catalog} state={builderState} selection={selection} disabled={authoringDisabled}
                        onUpdateNode={handleUpdateNode} onUpdateEdge={handleUpdateEdge} onRemoveNode={handleRemoveNode}
                        onRemoveEdge={handleRemoveEdge} onAddHitlGate={handleAddHitlGate}
                        onUpdateSettings={(patch) => updateState((previous) => ({ ...previous, extraConfig: { ...(previous.extraConfig || {}), ...patch } }))}
                      />
                    </AccordionDetails>
                  </Accordion>
                </Box>
              </Box>
            ) : null}
          </Box>
        )}
        <Dialog open={saveDialogOpen} onClose={() => busyAction ? undefined : setSaveDialogOpen(false)} fullWidth maxWidth="sm">
          <DialogTitle>Save Workflow</DialogTitle>
          <DialogContent dividers>
            <BuilderPersistencePanel
              form={persistenceForm}
              onFormChange={updatePersistenceForm}
              persisted={persistedWorkflow}
              busyAction={busyAction}
              statusMessage={persistenceStatus}
              errorMessage={persistenceError}
              canSave={Boolean(spec && persistenceForm.name.trim())}
              authoringDisabled={authoringDisabled}
              boundaryMessages={boundary.messages}
              onSave={handleSaveInternalWorkflow}
              onDelete={handleDeleteInternalWorkflow}
              showHeader={false}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setSaveDialogOpen(false)} disabled={Boolean(busyAction)}>Close</Button>
          </DialogActions>
        </Dialog>
        <Dialog open={Boolean(nodeRequest)} onClose={() => setNodeRequest(null)} fullWidth maxWidth="sm">
          <DialogTitle>
            {nodeRequest?.mode === 'before' ? 'Add a compatible previous step' : 'Add a compatible next step'}
          </DialogTitle>
          <DialogContent>
            {nodeRequest?.mode === 'after' && nodeRequest.route ? (
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                Connecting the “{nodeRequest.route.replace(/_/g, ' ')}” branch from {nodeRequest.source}.
              </Typography>
            ) : null}
            {nodeRequest?.mode === 'before' && nodeRequest.incomingPaths.length > 1 && !nodeRequest.selectedPathId ? (
              <>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  This step has multiple incoming paths. Choose the one where the new step should be inserted.
                </Typography>
                <List dense>
                  {nodeRequest.incomingPaths.map((path) => {
                    const edge = builderState?.edges[path.edgeIndex];
                    const source = path.source === 'START'
                      ? 'Start'
                      : builderState?.nodes.find((node) => node.id === path.source)?.label
                        || catalog?.node_catalog[builderState?.nodes.find((node) => node.id === path.source)?.type || '']?.display_name
                        || path.source;
                    const target = path.target === 'END'
                      ? 'End'
                      : builderState?.nodes.find((node) => node.id === path.target)?.label
                        || catalog?.node_catalog[builderState?.nodes.find((node) => node.id === path.target)?.type || '']?.display_name
                        || path.target;
                    const route = path.route
                      ? catalog?.route_functions[edge?.route_fn || '']?.route_options?.[path.route]?.display_name
                        || path.route.replace(/_/g, ' ')
                      : null;
                    return (
                      <ListItemButton
                        key={path.id}
                        onClick={() => setNodeRequest({ ...nodeRequest, selectedPathId: path.id })}
                      >
                        <ListItemText
                          primary={route ? `${source} — ${route} → ${target}` : `${source} → ${target}`}
                          secondary="Insert into this incoming path"
                        />
                      </ListItemButton>
                    );
                  })}
                </List>
              </>
            ) : null}
            {nodeRequest?.mode === 'before' && nodeRequest.selectedPathId && nodeRequest.incomingPaths.length > 1 ? (
              <Button
                size="small"
                onClick={() => setNodeRequest({ ...nodeRequest, selectedPathId: undefined })}
                sx={{ mb: 0.5 }}
              >
                Change incoming path
              </Button>
            ) : null}
            <List dense>
              {builderState && nodeRequest?.mode === 'after' ? builderState.nodes
                .filter((node) => node.id !== nodeRequest.source && canConnectNodes(catalog!, builderState, nodeRequest.source, node.id).ok)
                .map((node) => (
                  <ListItemButton key={node.id} onClick={() => {
                    handleConnectNodes(nodeRequest.source, node.id, nodeRequest.route);
                    setNodeRequest(null);
                  }}>
                    <ListItemText primary={node.label || catalog?.node_catalog[node.type]?.display_name || node.id} secondary="Connect existing node" />
                  </ListItemButton>
                )) : null}
              {builderState && nodeRequest?.mode === 'after' ? Object.entries(catalog?.node_catalog || {})
                .filter(([nodeType]) => canAddNodeType(catalog!, builderState, nodeType).ok)
                .filter(([nodeType]) => {
                  if (nodeRequest.source === 'START') {
                    return (catalog?.node_catalog[nodeType]?.allowed_parent_types || []).includes('START');
                  }
                  const sourceType = builderState.nodes.find((node) => node.id === nodeRequest.source)?.type;
                  return sourceType ? (catalog?.node_catalog[sourceType]?.allowed_child_types || []).includes(nodeType) : false;
                })
                .map(([nodeType, entry]) => (
                  <ListItemButton key={`new:${nodeType}`} onClick={() => {
                    const id = getCanonicalNodeId(nodeType, builderState.nodes.map((node) => node.id));
                    const allowedTools = getAllowedToolContractsForNode(catalog!, nodeType);
                    updateState((previous) => connectState({
                      ...previous,
                      nodes: [...previous.nodes, {
                        id,
                        type: nodeType,
                        ...(allowedTools[0] ? { tool_contract_ids: [allowedTools[0].id] } : {}),
                      }],
                    }, nodeRequest.source, id, nodeRequest.route));
                    setSelection({ kind: 'node', nodeId: id });
                    setNodeRequest(null);
                  }}>
                    <ListItemText primary={`Add ${entry.display_name || nodeType}`} secondary={entry.ui?.summary} />
                  </ListItemButton>
                )) : null}
              {builderState && nodeRequest?.mode === 'before'
                && (nodeRequest.incomingPaths.length === 0 || nodeRequest.selectedPathId)
                ? (() => {
                  const incomingPath = nodeRequest.incomingPaths.find((path) => path.id === nodeRequest.selectedPathId);
                  const existing = builderState.nodes
                    .filter((node) => canInsertExistingNodeBefore(
                      catalog!,
                      builderState,
                      nodeRequest.target,
                      node.id,
                      incomingPath,
                    ).ok)
                    .map((node) => (
                      <ListItemButton key={node.id} onClick={() => {
                        const position = positionForPreviousStep(builderState, nodeRequest.target, incomingPath);
                        updateState((previous) => {
                          const positioned = {
                            ...node,
                            position,
                          };
                          const withPosition = {
                            ...previous,
                            nodes: previous.nodes.map((candidate) => candidate.id === node.id ? positioned : candidate),
                          };
                          return insertNodeBefore(withPosition, nodeRequest.target, positioned, incomingPath, false);
                        });
                        setSelection({ kind: 'node', nodeId: node.id });
                        setNodeRequest(null);
                      }}>
                        <ListItemText
                          primary={node.label || catalog?.node_catalog[node.type]?.display_name || node.id}
                          secondary="Insert unconnected existing node"
                        />
                      </ListItemButton>
                    ));
                  const fresh = Object.entries(catalog?.node_catalog || {})
                    .filter(([nodeType]) => canInsertNodeTypeBefore(
                      catalog!,
                      builderState,
                      nodeRequest.target,
                      nodeType,
                      incomingPath,
                    ).ok)
                    .map(([nodeType, entry]) => (
                      <ListItemButton key={`new:${nodeType}`} onClick={() => {
                        const id = getCanonicalNodeId(nodeType, builderState.nodes.map((node) => node.id));
                        const allowedTools = getAllowedToolContractsForNode(catalog!, nodeType);
                        const nextNode: BuilderNodeState = {
                          id,
                          type: nodeType,
                          position: positionForPreviousStep(builderState, nodeRequest.target, incomingPath),
                          ...(allowedTools[0] ? { tool_contract_ids: [allowedTools[0].id] } : {}),
                        };
                        updateState((previous) => insertNodeBefore(
                          previous,
                          nodeRequest.target,
                          nextNode,
                          incomingPath,
                        ));
                        setSelection({ kind: 'node', nodeId: id });
                        setNodeRequest(null);
                      }}>
                        <ListItemText primary={`Add ${entry.display_name || nodeType}`} secondary={entry.ui?.summary} />
                      </ListItemButton>
                    ));
                  return [...existing, ...fresh];
                })()
                : null}
            </List>
            <Button onClick={() => setNodeRequest(null)}>Cancel</Button>
          </DialogContent>
        </Dialog>
      </Box>
    </ThemeProvider>
  );
}
