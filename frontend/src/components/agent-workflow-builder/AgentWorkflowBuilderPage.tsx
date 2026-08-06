import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  CssBaseline,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  List,
  ListItemButton,
  ListItemText,
  Typography,
} from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../../theme';
import { useAppThemeMode } from '../../hooks/useAppThemeMode';
import {
  saveInternalAgentWorkflow,
  deleteInternalAgentWorkflow,
  getInternalAgentWorkflow,
  getBuiltinAgentWorkflowSource,
  getInternalAgentWorkflowCatalog,
  listAgentWorkflows,
  validateAgentWorkflowSpec,
  type AgentWorkflow,
  type AgentWorkflowCatalogResponse,
  type AgentWorkflowValidationReport,
  type Thread,
} from '../../lib/api';
import {
  assembleAgentWorkflowSpec,
  AGENT_WORKFLOW_STARTER_WORKFLOW_IDS,
  canAddNodeType,
  canConnectNodes,
  canConnectNodeTypeToTarget,
  canConnectSourceToType,
  canInsertExistingNodeBefore,
  canInsertNodeTypeBefore,
  createHitlGateForTarget,
  getIncomingPaths,
  getAllowedToolContractsForNode,
  getCanonicalNodeId,
  getAllowedRouteFunctionsForNode,
  getRouteLabelsForFunction,
  insertNodeBefore,
  loadBuilderStateFromSpec,
  normalizeBuilderState,
  setHitlContinueWithoutTarget,
  type AgentWorkflowBuilderState,
  type AgentWorkflowStarter,
  type BuilderEdgeState,
  type BuilderIncomingPath,
  type BuilderNodeState,
} from '../../lib/agent-workflow-builder';
import { BuiltinAgentNodeType, RouteFunctionId } from '../../lib/enums';
import BuilderActionsBar from './BuilderActionsBar';
import BuilderGraphEditor from './BuilderGraphEditor';
import BuilderInspector from './BuilderInspector';
import BuilderNodePalette from './BuilderNodePalette';
import BuilderPersistencePanel, {
  type BuilderPersistedWorkflow,
  type BuilderPersistenceState,
} from './BuilderPersistencePanel';
import BuilderValidationPanel from './BuilderValidationPanel';
import BuilderSpecPanel from './BuilderSpecPanel';
import BuilderUtilityPanel from './BuilderUtilityPanel';
import type { BuilderSelection, BuilderValidationIssue } from './types';
import WorkbenchShell, { useWorkbenchLayout } from '../workbench/WorkbenchShell';
import DockMenuButton from '../workbench/DockMenuButton';
import WorkspaceTabs, { type WorkspaceTab } from '../workbench/WorkspaceTabs';
import ThreadWorkspaceContent from '../workbench/ThreadWorkspaceContent';
import useTraceTabs from '../workbench/useTraceTabs';
import useStoredLayoutState from '../workbench/useStoredLayoutState';
import type { ResolvedWorkbenchPlacement } from '../../lib/workbench-layout';
import ChatInterface, { type ChatTraceDescriptor } from '../ChatInterface';
import ThreadSecondaryPanel from '../ThreadSecondaryPanel';
import { buildDocumentWorkspaceTabs, type PdfTab } from '../../lib/document-tabs';
import { getThread } from '../../lib/api';
import { hydrateThreadPdfTab, loadThreadTabs } from '../../lib/thread-utils';
import { getActiveTab, getActiveTabData } from '../../lib/pdf-utils';
import { emptyBuilderTestSession, type BuilderTestSession } from '../../lib/builder-test-session';
import {
  BUILDER_LAYOUT_STORAGE_KEY,
  DEFAULT_BUILDER_LAYOUT,
  normalizeBuilderLayout,
} from '../../lib/builder-layout';

type ContextualNodeRequest =
  | { mode: 'after'; source: string; route?: string }
  | { mode: 'before'; target: string; incomingPaths: BuilderIncomingPath[]; selectedPathId?: string };

const collectNodeToolIds = (nodes: BuilderNodeState[]) => (
  Array.from(new Set(nodes.flatMap((node) => node.tool_contract_ids || []))).sort()
);

const BUILTIN_STARTERS: AgentWorkflowStarter[] = ['router', 'plan_execute', 'evaluator_replanner', 'orchestrator_worker'];

const isBuiltinStarter = (value: string): value is AgentWorkflowStarter => (
  BUILTIN_STARTERS.includes(value as AgentWorkflowStarter)
);

const customStarterValue = (workflowId: string) => `custom:${workflowId}`;

const builderStateFromWorkflowSpec = (
  catalog: AgentWorkflowCatalogResponse,
  spec: Record<string, any>,
): AgentWorkflowBuilderState => {
  const loadedState = normalizeBuilderState(catalog, loadBuilderStateFromSpec(spec));
  return { ...loadedState, allowed_tool_ids: collectNodeToolIds(loadedState.nodes) };
};

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

export default function AgentWorkflowBuilderPage() {
  const { darkMode, toggleDarkMode, hydrated: themeHydrated } = useAppThemeMode();
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
  const [persistenceForm, setPersistenceForm] = useState<BuilderPersistenceState>({
    workflowId: '',
    name: 'Internal Custom Agent',
    description: '',
  });
  const [persistedWorkflow, setPersistedWorkflow] = useState<BuilderPersistedWorkflow | null>(null);
  const [busyAction, setBusyAction] = useState<'save' | 'delete' | null>(null);
  const [persistenceStatus, setPersistenceStatus] = useState<string | null>(null);
  const [persistenceError, setPersistenceError] = useState<string | null>(null);
  const authoringDisabled = false;
  const undoStack = useRef<AgentWorkflowBuilderState[]>([]);
  const redoStack = useRef<AgentWorkflowBuilderState[]>([]);
  const [historyVersion, setHistoryVersion] = useState(0);
  const [isDirty, setIsDirty] = useState(false);
  const [nodeRequest, setNodeRequest] = useState<ContextualNodeRequest | null>(null);
  const [workspace, setWorkspace] = useState<'build' | 'test'>('build');
  const [buildTab, setBuildTab] = useState<'canvas-tab' | 'spec-tab'>('canvas-tab');
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [builderLayout, setBuilderLayout] = useStoredLayoutState(
    BUILDER_LAYOUT_STORAGE_KEY,
    DEFAULT_BUILDER_LAYOUT,
    normalizeBuilderLayout,
  );
  const utilityRailRef = useRef<HTMLDivElement | null>(null);
  const [buildWorkbenchLayout, setBuildWorkbenchLayout] = useWorkbenchLayout('askpdf.workbench.builder.build');
  const [testWorkbenchLayout, setTestWorkbenchLayout] = useWorkbenchLayout('askpdf.workbench.builder.test');
  const [resolvedPlacement, setResolvedPlacement] = useState<ResolvedWorkbenchPlacement>('right');
  const [isWorkbenchResizing, setIsWorkbenchResizing] = useState(false);
  const activeWorkbenchLayout = workspace === 'build' ? buildWorkbenchLayout : testWorkbenchLayout;
  const setActiveWorkbenchLayout = workspace === 'build' ? setBuildWorkbenchLayout : setTestWorkbenchLayout;
  const [testThread, setTestThread] = useState<Thread | null>(null);
  const [testPdfTabs, setTestPdfTabs] = useState<PdfTab[]>([]);
  const [testActiveTabId, setTestActiveTabId] = useState<string | null>(null);
  const [testThreadLoading, setTestThreadLoading] = useState(false);
  const [testSidebarVersion, setTestSidebarVersion] = useState(0);
  const {
    traceTabs: testTraceTabs,
    activeTraceId: testActiveTraceId,
    setActiveTraceId: setTestActiveTraceId,
    openTrace: openTestTrace,
    closeTrace: closeTestTrace,
    clearTraces: clearTestTraces,
  } = useTraceTabs();
  const [testSession, setTestSession] = useState<BuilderTestSession>(() => emptyBuilderTestSession());

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([
      getInternalAgentWorkflowCatalog(),
      listAgentWorkflows().catch(() => ({ agent_workflows: [] })),
      getBuiltinAgentWorkflowSource(AGENT_WORKFLOW_STARTER_WORKFLOW_IDS.router),
    ])
      .then(([nextCatalog, patternList, routerWorkflow]) => {
        if (cancelled) return;
        setCatalog(nextCatalog);
        setCustomWorkflows((patternList.agent_workflows || []).filter((pattern) => !pattern.is_builtin));
        setBuilderState(builderStateFromWorkflowSpec(nextCatalog, routerWorkflow.spec_json));
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

  const resetToStarter = useCallback(async (nextStarter: AgentWorkflowStarter = 'router') => {
    if (!catalog || authoringDisabled) return;
    try {
      const workflow = await getBuiltinAgentWorkflowSource(AGENT_WORKFLOW_STARTER_WORKFLOW_IDS[nextStarter]);
      setBuilderState(builderStateFromWorkflowSpec(catalog, workflow.spec_json));
      setStarter(nextStarter);
      setSelection(null);
      setValidation(null);
      setPersistedWorkflow(null);
      setPersistenceStatus(null);
      setPersistenceError(null);
      undoStack.current = [];
      redoStack.current = [];
      setHistoryVersion((value) => value + 1);
      setIsDirty(false);
    } catch (err) {
      setPersistenceError(err instanceof Error ? err.message : String(err));
    }
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
    void resetToStarter(nextStarter);
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
    const sourceType = previous.nodes.find((node) => node.id === source)?.type;
    const targetType = previous.nodes.find((node) => node.id === target)?.type;
    if (sourceType === BuiltinAgentNodeType.ParallelDispatch) {
      if (targetType === BuiltinAgentNodeType.Aggregator) {
        const existing = previous.edges.findIndex((edge) => edge.from === source && edge.conditional);
        const parallelEdge = {
          from: source,
          conditional: true,
          route_fn: RouteFunctionId.ParallelDispatch,
          routes: { dispatch: target },
        };
        return {
          ...previous,
          edges: existing >= 0
            ? previous.edges.map((edge, index) => index === existing ? parallelEdge : edge)
            : [...previous.edges, parallelEdge],
        };
      }
      const workerTypes = new Set([
        'retrieval_worker',
        'thread_conversation_history_worker',
        'durable_memory_worker',
        'thread_events_worker',
        'web_worker',
      ]);
      if (targetType && workerTypes.has(targetType)) {
        const duplicate = previous.edges.some((edge) => edge.dynamic && edge.from === source && edge.to === target);
        return duplicate ? previous : { ...previous, edges: [...previous.edges, { from: source, to: target, dynamic: true }] };
      }
    }
    if (route) {
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

  const handleUpdateHitlBypass = (gateNodeId: string, targetId?: string) => {
    updateState((previous) => setHitlContinueWithoutTarget(previous, gateNodeId, targetId));
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
  const workflowIsValid = Boolean(validation?.valid) && !validationIssues.some((issue) => issue.severity === 'error');
  const testDisabledReason = !validation
    ? 'Validate the workflow before testing'
    : workflowIsValid
      ? undefined
      : 'Fix validation errors before testing';
  const baseWorkflowId = workflowIdFromCustomStarter(starter)
    || AGENT_WORKFLOW_STARTER_WORKFLOW_IDS[starter as AgentWorkflowStarter]
    || String(spec?.workflow_id || 'router_rag_agent');
  const previousTestWorkflowId = useRef(baseWorkflowId);
  useEffect(() => {
    if (previousTestWorkflowId.current !== baseWorkflowId) {
      previousTestWorkflowId.current = baseWorkflowId;
      setTestSession(emptyBuilderTestSession(testThread?.id || null));
      clearTestTraces();
    }
  }, [baseWorkflowId, clearTestTraces, testThread?.id]);

  const testActiveDocument = getActiveTab(testPdfTabs, testActiveTabId);
  const testActiveDocumentData = getActiveTabData(testActiveDocument);
  const testWorkspaceTabs = useMemo(() => buildDocumentWorkspaceTabs({
    enabled: Boolean(testThread),
    documents: testPdfTabs,
    traces: testTraceTabs,
  }), [testPdfTabs, testThread, testTraceTabs]);

  const handleTestThreadSelect = useCallback(async (thread: Thread | null) => {
    if (!thread) {
      setTestThread(null);
      setTestPdfTabs([]);
      setTestActiveTabId(null);
      setTestSession(emptyBuilderTestSession());
      clearTestTraces();
      return;
    }
    setTestThreadLoading(true);
    try {
      const detailed = await getThread(thread.id);
      const tabs = await loadThreadTabs(detailed);
      setTestThread(detailed);
      setTestPdfTabs(tabs);
      setTestActiveTabId(tabs[0]?.id || 'browser-tab');
      setTestSession(emptyBuilderTestSession(detailed.id));
      clearTestTraces();
      window.setTimeout(() => {
        detailed.files.slice(1).forEach(async (threadFile) => {
          try {
            const hydrated = await hydrateThreadPdfTab(detailed.id, threadFile);
            setTestPdfTabs(prev => prev.map(tab => tab.fileHash === hydrated.fileHash ? hydrated : tab));
          } catch (error) {
            console.warn(`Failed to hydrate background test PDF tab ${threadFile.fileHash}:`, error);
          }
        });
      }, 0);
    } finally {
      setTestThreadLoading(false);
    }
  }, [clearTestTraces]);

  const handleTestTabChange = useCallback((tabId: string) => {
    setTestActiveTabId(tabId);
    const tab = testPdfTabs.find(item => item.id === tabId);
    if (!testThread || !tab || tabId === 'browser-tab' || tab.sentences) return;
    void hydrateThreadPdfTab(testThread.id, {
      fileHash: tab.fileHash,
      fileName: tab.fileName,
      sourceType: tab.sourceType,
    }).then((hydrated) => {
      setTestPdfTabs(prev => prev.map(item => item.fileHash === hydrated.fileHash ? hydrated : item));
    }).catch((error) => {
      console.warn(`Failed to hydrate selected test PDF tab ${tab.fileHash}:`, error);
    });
  }, [testPdfTabs, testThread]);

  const handleOpenTestTrace = useCallback((trace: ChatTraceDescriptor) => {
    openTestTrace(trace);
    setTestActiveTabId('trace-tab');
  }, [openTestTrace]);

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

  if (!themeHydrated) return null;

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', bgcolor: 'background.default', overflow: 'hidden' }}>
        <WorkbenchShell
          layout={activeWorkbenchLayout}
          onLayoutChange={setActiveWorkbenchLayout}
          onResolvedPlacementChange={setResolvedPlacement}
          onResizingChange={setIsWorkbenchResizing}
          secondaryLabel={workspace === 'build' ? 'Node Inspector and Node Palette' : 'Workflow test controls'}
          primaryToolbar={
            <BuilderActionsBar
              starter={starter}
              customWorkflows={customWorkflows}
              disabled={authoringDisabled}
              onStarterChange={handleStarterChange}
              onReset={() => resetToStarter()}
              onValidate={handleValidate}
              validating={validating}
              dirty={isDirty}
              canUndo={undoStack.current.length > 0}
              canRedo={redoStack.current.length > 0}
              onUndo={handleUndo}
              onRedo={handleRedo}
              onOpenSave={() => setSaveDialogOpen(true)}
              saveBusy={busyAction === 'save'}
              savedWorkflowId={persistedWorkflow?.workflow.id}
              workflowName={persistenceForm.name}
              testMode={workspace === 'test'}
              onToggleTest={() => setWorkspace((current) => current === 'build' && !workflowIsValid ? current : current === 'build' ? 'test' : 'build')}
              testDisabled={!workflowIsValid}
              testDisabledReason={testDisabledReason}
              hasTestSession={testSession.messages.length > 0}
              onClearTestSession={() => {
                setTestSession(emptyBuilderTestSession(testThread?.id || null));
                clearTestTraces();
              }}
              darkMode={darkMode}
              onToggleDarkMode={toggleDarkMode}
              onGoHome={() => { window.location.assign('/'); }}
              layoutControl={
                <DockMenuButton
                  value={activeWorkbenchLayout}
                  resolvedPlacement={resolvedPlacement}
                  onChange={setActiveWorkbenchLayout}
                  label={workspace === 'build' ? 'Builder utilities layout' : 'Test conversation layout'}
                />
              }
            />
          }
          primaryTabs={
            workspace === 'build' ? (
              <WorkspaceTabs
                tabs={[
                  { kind: 'canvas', id: 'canvas-tab', label: 'Canvas', issueCount: validationIssues.length },
                  { kind: 'spec', id: 'spec-tab', label: 'Spec', dirty: isDirty },
                ] satisfies WorkspaceTab[]}
                activeTabId={buildTab}
                onTabChange={(tabId) => setBuildTab(tabId as 'canvas-tab' | 'spec-tab')}
              />
            ) : testThread ? (
              <WorkspaceTabs
                tabs={testWorkspaceTabs}
                activeTabId={testActiveTabId}
                onTabChange={handleTestTabChange}
              />
            ) : null
          }
          primaryContent={
            loading ? (
              <Box sx={{ height: '100%', display: 'grid', placeItems: 'center' }}><CircularProgress /></Box>
            ) : error || !catalog || !builderState ? (
              <Box sx={{ p: 2 }}><Alert severity="error">{error || 'Agent workflow catalog is unavailable.'}</Alert></Box>
            ) : workspace === 'test' ? (
              testThreadLoading ? (
                <Box sx={{ height: '100%', display: 'grid', placeItems: 'center' }}><CircularProgress /></Box>
              ) : (
                <ThreadWorkspaceContent
                  activeTabId={testActiveTabId}
                  activeDocument={testActiveDocument}
                  documentSentences={testActiveDocumentData.pdfSentences}
                  documentDownloadUrl={testActiveDocumentData.downloadUrl}
                  traceTabs={testTraceTabs}
                  activeTraceId={testActiveTraceId}
                  onActiveTraceChange={setTestActiveTraceId}
                  onCloseTrace={closeTestTrace}
                  isResizing={isWorkbenchResizing}
                  darkMode={darkMode}
                  currentDocumentSentenceId={null}
                  onDocumentJump={() => undefined}
                  autoScroll={false}
                  highlightEnabled
                  threadId={testThread?.id || null}
                  emptyTitle={testThread ? 'Choose a workspace tab' : 'Select a thread to test'}
                  emptyDescription={testThread ? 'Open a PDF, Browser, or Debug Trace.' : 'Use the project-grouped thread browser in the secondary panel.'}
                />
              )
            ) : buildTab === 'spec-tab' && spec ? (
              <BuilderSpecPanel spec={spec} />
            ) : (
              <Box sx={{ height: '100%', minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column', bgcolor: 'background.default' }}>
                {spec && (
                  <BuilderValidationPanel
                    validation={validation}
                    issues={validationIssues}
                    workflowIsValid={workflowIsValid}
                    onSelectIssue={(nextSelection) => { setSelection(nextSelection); setBuildTab('canvas-tab'); }}
                  />
                )}
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
                  graphElementsRatio={builderLayout.graphElementsRatio}
                  graphElementsCollapsed={builderLayout.graphElementsCollapsed}
                  onGraphElementsRatioChange={(graphElementsRatio) => setBuilderLayout((previous) => ({ ...previous, graphElementsRatio }))}
                  onGraphElementsCollapsedChange={(graphElementsCollapsed) => setBuilderLayout((previous) => ({ ...previous, graphElementsCollapsed }))}
                />
              </Box>
            )
          }
          secondaryHeader={workspace === 'build' ? (
            <Box sx={{ minHeight: 44, px: 1.5, display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: 1, borderColor: 'divider' }}>
              <Box sx={{ minWidth: 0, display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>Agent Workflow Builder</Typography>
                {builderState ? (
                  <Chip
                    size="small"
                    variant="outlined"
                    label={`${builderState.nodes.length} nodes · ${builderState.edges.length} edges · ${builderState.allowed_tool_ids.length} tools`}
                    sx={{ height: 22, fontSize: '0.68rem' }}
                  />
                ) : null}
              </Box>
            </Box>
          ) : undefined}
          secondaryContent={
            !catalog || !builderState ? <Box /> : workspace === 'test' ? (
              <ThreadSecondaryPanel
                activeThread={testThread}
                sidebarKey={testSidebarVersion}
                selectionOnly
                activeThreadId={null}
                onThreadSelect={(thread) => void handleTestThreadSelect(thread)}
                onBackToProject={() => void handleTestThreadSelect(null)}
                darkMode={darkMode}
                renderSelectedTitle={(thread) => (
                  <Typography variant="subtitle2" noWrap title={thread.name || thread.id}>
                    {thread.name || thread.id}
                  </Typography>
                )}
                renderConversation={(thread) => spec ? (
                  <ChatInterface
                    activeThread={thread}
                    chatSentences={[]}
                    setChatSentences={() => undefined}
                    currentChatId={null}
                    activeSource="chat"
                    onJump={() => undefined}
                    onOpenTrace={handleOpenTestTrace}
                    darkMode={darkMode}
                    testRuntime={{
                      kind: 'builder-test',
                      persistent: false,
                      historyReadOnly: true,
                      spec,
                      baseWorkflowId,
                      session: testSession,
                      onSessionChange: setTestSession,
                    }}
                  />
                ) : null}
              />
            ) : (
              <BuilderUtilityPanel
                placement={resolvedPlacement}
                utilityRailRef={utilityRailRef}
                selectionKey={
                  selection?.kind === 'node'
                    ? `node:${selection.nodeId}`
                    : selection?.kind === 'edge'
                      ? `edge:${selection.edgeIndex}`
                      : null
                }
                inspector={
                  <BuilderInspector
                    catalog={catalog} state={builderState} selection={selection} disabled={authoringDisabled}
                    onUpdateNode={handleUpdateNode} onUpdateHitlBypass={handleUpdateHitlBypass} onUpdateEdge={handleUpdateEdge} onRemoveNode={handleRemoveNode}
                    onRemoveEdge={handleRemoveEdge} onAddHitlGate={handleAddHitlGate}
                    onUpdateSettings={(patch) => updateState((previous) => ({ ...previous, extraConfig: { ...(previous.extraConfig || {}), ...patch } }))}
                  />
                }
                palette={<BuilderNodePalette catalog={catalog} state={builderState} disabled={authoringDisabled} onAddNodeType={handleAddNodeType} onAddNote={handleAddNote} onAddGroup={handleAddGroup} />}
              />
            )
          }
        />
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
              boundaryMessages={[]}
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
              {builderState && nodeRequest?.mode === 'after' ? (
                <Box
                  component="li"
                  sx={{ borderTop: 1, borderColor: 'divider', listStyle: 'none', mt: 0.5, pt: 1, px: 2 }}
                >
                  <Typography variant="overline" color="text.secondary">
                    Add a directly compatible node
                  </Typography>
                </Box>
              ) : null}
              {builderState && nodeRequest?.mode === 'after' ? Object.entries(catalog?.node_catalog || {})
                .filter(([nodeType]) => canAddNodeType(catalog!, builderState, nodeType).ok)
                .filter(([nodeType]) => canConnectSourceToType(catalog!, builderState, nodeRequest.source, nodeType).ok)
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
                  const hitlGateEntry = catalog?.node_catalog[BuiltinAgentNodeType.HitlGate];
                  const hitlGateId = getCanonicalNodeId(
                    `hitl_${nodeRequest.target}`,
                    builderState.nodes.map((node) => node.id),
                  );
                  const hitlGate = hitlGateEntry
                    && canAddNodeType(catalog!, builderState, BuiltinAgentNodeType.HitlGate).ok
                    ? (
                      <ListItemButton key={`new:${BuiltinAgentNodeType.HitlGate}`} onClick={() => {
                        updateState((previous) => createHitlGateForTarget(
                          catalog!,
                          previous,
                          nodeRequest.target,
                          {
                            id: hitlGateId,
                            incomingPath,
                            position: positionForPreviousStep(
                              builderState,
                              nodeRequest.target,
                              incomingPath,
                            ),
                          },
                        ));
                        setSelection({ kind: 'node', nodeId: hitlGateId });
                        setNodeRequest(null);
                      }}>
                        <ListItemText
                          primary={`Add ${hitlGateEntry.display_name || 'Human Review Gate'}`}
                          secondary="Pause for human approval before this step"
                        />
                      </ListItemButton>
                    )
                    : null;
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
                    .filter(([nodeType]) => nodeType !== BuiltinAgentNodeType.HitlGate)
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
                  const strictNodeTypes = new Set(
                    Object.entries(catalog?.node_catalog || {})
                      .filter(([nodeType]) => canInsertNodeTypeBefore(
                        catalog!,
                        builderState,
                        nodeRequest.target,
                        nodeType,
                        incomingPath,
                      ).ok)
                      .map(([nodeType]) => nodeType),
                  );
                  const direct = Object.entries(catalog?.node_catalog || {})
                    .filter(([nodeType]) => !strictNodeTypes.has(nodeType))
                    .filter(([nodeType]) => canAddNodeType(catalog!, builderState, nodeType).ok)
                    .filter(([nodeType]) => getAllowedRouteFunctionsForNode(catalog!, nodeType).length === 0)
                    .filter(([nodeType]) => canConnectNodeTypeToTarget(
                      catalog!,
                      builderState,
                      nodeType,
                      nodeRequest.target,
                    ).ok)
                    .map(([nodeType, entry]) => (
                      <ListItemButton key={`direct:${nodeType}`} onClick={() => {
                        const id = getCanonicalNodeId(nodeType, builderState.nodes.map((node) => node.id));
                        const allowedTools = getAllowedToolContractsForNode(catalog!, nodeType);
                        const nextNode: BuilderNodeState = {
                          id,
                          type: nodeType,
                          position: positionForPreviousStep(builderState, nodeRequest.target),
                          ...(allowedTools[0] ? { tool_contract_ids: [allowedTools[0].id] } : {}),
                        };
                        updateState((previous) => insertNodeBefore(
                          previous,
                          nodeRequest.target,
                          nextNode,
                        ));
                        setSelection({ kind: 'node', nodeId: id });
                        setNodeRequest(null);
                      }}>
                        <ListItemText
                          primary={`Add ${entry.display_name || nodeType}`}
                          secondary="Connect directly to this step and keep its existing incoming path"
                        />
                      </ListItemButton>
                    ));
                  return (
                    <>
                      {existing}
                      {hitlGate}
                      {fresh}
                      {direct.length > 0 ? (
                        <Box
                          component="li"
                          sx={{ borderTop: 1, borderColor: 'divider', listStyle: 'none', mt: 0.5, pt: 1, px: 2 }}
                        >
                          <Typography variant="overline" color="text.secondary">
                            Also compatible directly
                          </Typography>
                        </Box>
                      ) : null}
                      {direct}
                    </>
                  );
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
