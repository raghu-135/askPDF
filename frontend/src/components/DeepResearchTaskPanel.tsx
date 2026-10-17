import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert, Box, Button, Chip,
  Divider, IconButton, LinearProgress, ListItemIcon, ListItemText,
  Menu, MenuItem, Paper, Stack, TextField, Tooltip, Typography, useMediaQuery, useTheme,
} from '@mui/material';
import DashboardOutlinedIcon from '@mui/icons-material/DashboardOutlined';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import DeleteIcon from '@mui/icons-material/Delete';
import NavigateBeforeIcon from '@mui/icons-material/NavigateBefore';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import BugReportIcon from '@mui/icons-material/BugReport';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import TravelExploreIcon from '@mui/icons-material/TravelExplore';
import { packingBudgetPayload, type ContextBudgetRatios } from '../lib/context-budget';
import {
  API_BASE,
  commandAgentTask,
  createAgentTask,
  deleteAgentTask,
  downloadAgentTaskArtifact,
  listAgentDefinitions,
  getAgentRun,
  getAgentTask,
  getAgentTaskRuns,
  getAgentTaskTodos,
  getAgentTaskTimeline,
  listAgentTasks,
  resumeAgentRun,
  respondToAgentTaskResultReview,
  respondToAgentTaskBudgetReview,
  submitAgentTaskCourseCorrection,
  sendAgentRunFollowup,
  interruptAgentRunWithInput,
  steerAgentRunLive,
  type AgentTaskRun,
  type AgentTaskResearchCanvasRef,
  type AgentTaskSummary,
  type AgentTaskTimelineItem,
  type AgentTaskTodo,
  type AgentRunResumeAction,
  type AgentDefinitionCatalogEntry,
} from '../lib/api';
import {
  isRunOwnedBySelectedTask,
  isTerminalAgentTaskEvent,
  mergeActiveAgentTaskRun,
  selectAgentTaskRunIndex,
  shouldPollAgentTask,
  shouldRefreshAgentTaskTimeline,
  shouldSubscribeToAgentTaskEvents,
} from '../lib/deep-research-ui-state';
import { isRuntimeOperationEnabled, runtimeCapabilityResponseMatchesRun, runtimeInterruptResponseOperation, runtimeOperationAvailability, TASK_CONTROL_CATALOG } from '../lib/runtime-capabilities';
import { useAgentRunCapabilities } from '../lib/use-agent-run-capabilities';
import {
  deriveConversationSentences,
  type ConversationSentence,
  type ConversationSentenceCache,
} from '../lib/chat-sentence-cache';
import type { ChatTraceDescriptor } from './ChatInterface';
import { AGENT_SSE_RECONNECT_INTERVAL_MS, AGENT_TASK_POLL_INTERVAL_MS } from '../lib/agent-ui-config';
import {
  ConversationComposer,
  ConversationArtifactList,
  ConversationDisclosure,
  ConversationHeader,
  ConversationMarkdown,
  ConversationMessageActions,
  HumanReviewDecisionPanel,
  ConversationMessageBubble,
  ConversationPanelTemplate,
  ConversationTranscriptFrame,
  ResizableDecisionPanel,
  SourceList,
} from './conversation';
import {
  DeepResearchBudgetSummary,
  DeepResearchDecisionLead,
  DeepResearchDecisionStickyFooter,
  DeepResearchGuidanceField,
  DeepResearchIssueList,
  DeepResearchProvisionalAnswer,
} from './conversation/DeepResearchDecisionContent';
import {
  budgetUsageFromTask,
  countReviewIssues,
  effectiveObjectiveRequestsLiveWeb,
  resultReviewContinueLabel,
} from '../lib/deep-research-decision-ui';
import { citationSourcesFromTimelineItems } from '../lib/research-citation-display';
import type { CitationDisplaySource } from '../lib/research-citation-display';
import { ConversationMarkdownWithCitations } from './conversation/ConversationMarkdownWithCitations';
import { DeepResearchBudgetChips } from './conversation/DeepResearchBudgetChips';
import { formatAgentTaskStatus, formatTimelineItemStatus } from '../lib/deep-research-task-status';
import { WorkbenchSelect } from './workbench/WorkbenchToolbar';

export function DeepResearchTaskPicker({
  threadId,
  selectedTaskId,
  onSelect,
}: {
  threadId: string;
  selectedTaskId?: string | null;
  onSelect: (taskId: string | null) => void;
}) {
  const [anchor, setAnchor] = useState<HTMLElement | null>(null);
  const [tasks, setTasks] = useState<AgentTaskSummary[]>([]);
  const [busy, setBusy] = useState(false);
  const selected = tasks.find((task) => task.id === selectedTaskId);

  const refresh = useCallback(async () => setTasks(await listAgentTasks(threadId)), [threadId]);
  useEffect(() => { void refresh(); }, [refresh]);

  const remove = async (event: React.MouseEvent, task: AgentTaskSummary) => {
    event.stopPropagation();
    if (!window.confirm(`Delete Deep Research task “${task.objective}”?`)) return;
    setBusy(true);
    try {
      await deleteAgentTask(task.id, threadId, task.version);
      if (selectedTaskId === task.id) onSelect(null);
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  const taskPickerSecondary = (task: AgentTaskSummary) => {
    const status = `${task.status.replaceAll('_', ' ')} · attempt ${task.run_attempt || 0}`;
    const canvases = task.research_canvases || [];
    if (!canvases.length) return status;
    const lead = canvases.find((canvas) => canvas.is_current_on_thread) || canvases[0];
    const canvasHint = canvases.length === 1
      ? `Canvas: ${lead.title}`
      : `${canvases.length} canvases · ${lead.title}`;
    return `${status} · ${canvasHint}`;
  };

  return <>
    <Tooltip title="Deep Research tasks (long-running objectives separate from thread agent chat)">
      <IconButton
        size="small"
        color="inherit"
        aria-label="Deep Research"
        onClick={(event) => { setAnchor(event.currentTarget); void refresh(); }}
      >
        <TravelExploreIcon fontSize="small" />
      </IconButton>
    </Tooltip>
    <Menu anchorEl={anchor} open={Boolean(anchor)} onClose={() => setAnchor(null)} slotProps={{ paper: { sx: { width: 360, maxWidth: '90vw' } } }}>
      <MenuItem onClick={() => { onSelect(null); setAnchor(null); }}>
        <ListItemIcon><TravelExploreIcon fontSize="small" /></ListItemIcon>
        <ListItemText primary="New Deep Research task" secondary="Start a separate long-running research run" />
      </MenuItem>
      <Divider />
      {tasks.map((task) => <MenuItem key={task.id} selected={task.id === selectedTaskId} onClick={() => { onSelect(task.id); setAnchor(null); }}>
        <ListItemText
          primary={task.objective}
          secondary={taskPickerSecondary(task)}
          slotProps={{ primary: { noWrap: true }, secondary: { noWrap: true } }}
        />
        {['completed', 'failed', 'expired', 'cancelled', 'recovery_required'].includes(task.status) && <IconButton size="small" color="error" disabled={busy} onClick={(event) => void remove(event, task)} aria-label="Delete task">
          <DeleteIcon fontSize="small" />
        </IconButton>}
      </MenuItem>)}
      {!tasks.length && <MenuItem disabled>No research tasks yet</MenuItem>}
    </Menu>
  </>;
}


function DeepResearchTaskDeliverables({
  canvases,
  selectedRunId,
  onOpenCanvas,
}: {
  canvases: AgentTaskResearchCanvasRef[];
  selectedRunId?: string | null;
  onOpenCanvas?: (canvas: { id: string; title: string }) => void;
}) {
  if (!canvases.length || !onOpenCanvas) return null;
  return (
    <Paper variant="outlined" sx={{ mb: 1, p: 1.25, borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 0.75 }}>
        Research canvases from this task
      </Typography>
      <Stack spacing={0.75}>
        {canvases.map((canvas) => {
          const forCurrentRun = Boolean(selectedRunId && canvas.run_id === selectedRunId);
          return (
            <Box
              key={canvas.id}
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                p: 0.5,
                borderRadius: 1,
                bgcolor: forCurrentRun ? 'action.selected' : 'transparent',
              }}
            >
              <DashboardOutlinedIcon fontSize="small" color="primary" />
              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Typography variant="body2" noWrap sx={{ fontWeight: forCurrentRun ? 600 : 500 }}>
                  {canvas.title}
                </Typography>
                <Typography variant="caption" color="text.secondary" component="div">
                  {canvas.attempt != null ? `Attempt ${canvas.attempt}` : 'Research run'}
                  {canvas.is_current_on_thread ? ' · Current on thread' : ' · Older revision'}
                </Typography>
              </Box>
              <Button
                size="small"
                sx={{ textTransform: 'none', flexShrink: 0 }}
                onClick={() => onOpenCanvas({ id: canvas.id, title: canvas.title })}
              >
                Open canvas
              </Button>
            </Box>
          );
        })}
      </Stack>
    </Paper>
  );
}


function TimelineBubble({
  item,
  taskId,
  threadId,
  onSaveToMemory,
  onOpenCanvas,
  copied,
  active,
  onCopy,
  onReadAloud,
  rootRef,
  onSelectAttempt,
  citationSources,
}: {
  item: AgentTaskTimelineItem;
  taskId: string;
  threadId: string;
  onSaveToMemory?: (content: string) => void;
  onOpenCanvas?: (canvas: { id: string; title: string }) => void;
  copied: boolean;
  active: boolean;
  onCopy: () => void;
  onReadAloud: () => void;
  rootRef: React.Ref<HTMLLIElement>;
  onSelectAttempt?: (attempt: number) => void;
  citationSources?: CitationDisplaySource[];
}) {
  const isObjective = item.type === 'objective';
  const isGuidance = item.type === 'research_guidance';
  const isUserBubble = isObjective || isGuidance;
  const useCitationMarkdown = item.type === 'research_report' || item.type === 'todo_result';
  const foldEntries = Object.entries(item.folds || {}).filter(([label, value]) => (
    !['sources', 'artifacts'].includes(label)
    && value != null
    && (typeof value !== 'object' || Object.keys(value).length)
  ));
  const sourceGroups = [
    ['web', 'Web sources'],
    ['document', 'Document evidence'],
    ['memory', 'Memory context'],
    ['thread', 'Thread context'],
  ] as const;
  const badgeLabel = isGuidance
    ? `your guidance · ${formatTimelineItemStatus(item.status)}`
    : `${item.type.replaceAll('_', ' ')} · ${formatTimelineItemStatus(item.status)}`;
  const markdownBody = useCitationMarkdown
    ? <ConversationMarkdownWithCitations content={item.primary_content} citationSources={citationSources} />
    : undefined;
  return <ConversationMessageBubble
    rootRef={rootRef}
    role={isUserBubble ? 'user' : 'assistant'}
    content={markdownBody ? '' : item.primary_content}
    beforeContent={markdownBody}
    active={active}
    wide={!isUserBubble}
    badge={<Chip size="small" label={badgeLabel} sx={{ mb: 1 }} color={['todo_failure', 'run_failure'].includes(item.type) ? 'error' : item.type === 'research_canvas' || item.type === 'research_report' ? 'success' : isGuidance ? 'info' : 'default'} />}
    actions={<ConversationMessageActions copied={copied} readActive={active} onCopy={onCopy} onReadAloud={onReadAloud}>
      {onSaveToMemory && item.type === 'todo_result' && <Button size="small" onClick={() => onSaveToMemory(item.primary_content)}>Save to memory</Button>}
      {item.type === 'research_canvas' && onOpenCanvas && item.canvas_ref && (
        <Button size="small" onClick={() => onOpenCanvas(item.canvas_ref!)}>
          Open canvas
        </Button>
      )}
    </ConversationMessageActions>}
    afterContent={foldEntries.length || (item.sources || []).length || (item.artifacts || []).length ? <Box sx={{ mt: 1 }}>
      {sourceGroups.map(([kind, label]) => {
        const sources = (item.sources || []).filter((source) => source.kind === kind);
        if (!sources.length) return null;
        return <Box key={kind}>
          <SourceList label={label} sources={sources} />
          {[...new Set(sources.flatMap((source) => source.origins || []).filter((origin) => origin.inherited && origin.attempt).map((origin) => origin.attempt))].map((attempt) => (
            <Button key={`${kind}:${attempt}`} size="small" onClick={() => onSelectAttempt?.(attempt)}>View attempt {attempt}</Button>
          ))}
        </Box>;
      })}
      <ConversationArtifactList
        artifacts={item.artifacts || []}
        onDownload={(artifact) => void downloadAgentTaskArtifact(taskId, artifact.id, threadId)}
      />
      {foldEntries.map(([label, value]) => <ConversationDisclosure key={label} label={label.replaceAll('_', ' ')}>
        <Typography component="pre" variant="caption" sx={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{typeof value === 'string' ? value : JSON.stringify(value, null, 2)}</Typography>
      </ConversationDisclosure>)}
    </Box> : undefined}
  />;
}


export default function DeepResearchTaskPanel({
  threadId,
  models,
  model,
  contextWindow,
  contextBudget,
  defaultContextBudget,
  selectedTaskId,
  embeddingControl,
  renderWebControl,
  webSearchMode,
  onModelChange,
  onContextWindowChange,
  onContextBudgetChange,
  contextBudgetDisabled = false,
  onTaskSelect,
  onBack,
  onOpenTrace,
  onSaveToMemory,
  onOpenCanvas,
  chatSentences,
  setChatSentences,
  setChatPlaybackSourceKey,
  currentChatId,
  activeSource,
  onJump,
  autoScroll,
}: {
  threadId: string;
  models: string[];
  model: string;
  contextWindow: number;
  contextBudget: ContextBudgetRatios | null;
  defaultContextBudget: ContextBudgetRatios | null;
  contextBudgetDisabled?: boolean;
  selectedTaskId: string | null;
  embeddingControl: React.ReactNode;
  renderWebControl: (mode: 'off' | 'ask' | 'on', disabled: boolean) => React.ReactNode;
  webSearchMode: 'off' | 'ask' | 'on';
  onModelChange: (model: string) => void;
  onContextWindowChange: (value: number) => void;
  onContextBudgetChange: (value: ContextBudgetRatios) => void;
  onTaskSelect: (taskId: string | null) => void;
  onBack: () => void;
  onOpenTrace?: (trace: ChatTraceDescriptor) => void;
  onSaveToMemory?: (content: string) => void;
  onOpenCanvas?: (canvas: { id: string; title: string }) => void;
  chatSentences: ConversationSentence[];
  setChatSentences: (sentences: ConversationSentence[]) => void;
  setChatPlaybackSourceKey: (sourceKey: string) => void;
  currentChatId: number | null;
  activeSource: 'pdf' | 'chat';
  onJump: (id: number) => void;
  autoScroll: boolean;
}) {
  const [task, setTask] = useState<AgentTaskSummary | null>(null);
  const [runs, setRuns] = useState<AgentTaskRun[]>([]);
  const [todos, setTodos] = useState<AgentTaskTodo[]>([]);
  const [runIndex, setRunIndex] = useState(-1);
  const [items, setItems] = useState<AgentTaskTimelineItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [decisionSubmitting, setDecisionSubmitting] = useState<AgentRunResumeAction | null>(null);
  const [reviewGuidance, setReviewGuidance] = useState('');
  const [courseCorrection, setCourseCorrection] = useState('');
  const [courseCorrectionStatus, setCourseCorrectionStatus] = useState('');
  const [decisionError, setDecisionError] = useState('');
  const [error, setError] = useState('');
  const [definitions, setDefinitions] = useState<AgentDefinitionCatalogEntry[]>([]);
  const [definitionId, setDefinitionId] = useState('');
  const [deepResearchDiscoveryError, setDeepResearchDiscoveryError] = useState('');
  const [interactionOperation, setInteractionOperation] = useState<'run.send_followup' | 'run.interrupt_with_input' | 'run.steer_live'>('run.send_followup');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const lastSequence = useRef(0);
  const sequenceRunId = useRef<string | null>(null);
  const taskContextRef = useRef(selectedTaskId);
  taskContextRef.current = selectedTaskId;
  const sentenceCacheRef = useRef<ConversationSentenceCache>(new Map());
  const itemRefs = useRef(new Map<string, HTMLLIElement>());
  const panelRef = useRef<HTMLDivElement>(null);
  const selectedRun = runs[runIndex] || null;
  const selectedCapabilitiesState = useAgentRunCapabilities(selectedRun?.id, threadId, `${selectedRun?.status}:${selectedRun?.runtime_binding_status}:${selectedRun?.pending_interrupt?.interrupt_id}:${selectedRun?.pending_interrupt?.status}:${selectedRun?.pending_interrupt?.resume_version}:${task?.version}`);
  const activeCapabilitiesState = useAgentRunCapabilities(
    task?.active_run_id,
    threadId,
    `${task?.status}:${task?.version}:${task?.active_run?.runtime_binding_status}:${task?.active_run?.pending_interrupt?.interrupt_id}:${task?.active_run?.pending_interrupt?.status}`,
  );
  const selectedRunCapabilities = selectedCapabilitiesState.capabilities;
  const activeTaskCapabilities = runtimeCapabilityResponseMatchesRun(
    activeCapabilitiesState.capabilities, task?.active_run_id || '',
  ) ? activeCapabilitiesState.capabilities : null;
  const runtimeControlError = selectedCapabilitiesState.error || activeCapabilitiesState.error || '';

  useEffect(() => {
    setChatPlaybackSourceKey(`deep-research:${selectedTaskId || 'new'}:${selectedRun?.id || 'none'}`);
    setChatSentences(deriveConversationSentences(
      items.map((item) => ({ id: item.id, content: item.primary_content })),
      sentenceCacheRef.current,
    ));
  }, [items, selectedRun?.id, selectedTaskId, setChatPlaybackSourceKey, setChatSentences]);

  const activeItemId = activeSource === 'chat' && currentChatId !== null
    ? chatSentences[currentChatId]?.itemId || null
    : null;

  useEffect(() => {
    if (!autoScroll || !activeItemId) return;
    itemRefs.current.get(activeItemId)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, [activeItemId, autoScroll, currentChatId]);

  const copyItem = useCallback((item: AgentTaskTimelineItem) => {
    void navigator.clipboard.writeText(item.primary_content);
    setCopiedId(item.id);
    window.setTimeout(() => setCopiedId((current) => current === item.id ? null : current), 2000);
  }, []);

  const readItem = useCallback((itemId: string) => {
    const firstSentence = chatSentences.find((sentence) => sentence.itemId === itemId);
    if (firstSentence) onJump(firstSentence.id);
  }, [chatSentences, onJump]);

  useEffect(() => {
    let active = true;
    setDeepResearchDiscoveryError('');
    void listAgentDefinitions()
      .then((catalog) => {
        if (!active) return;
        const eligible = catalog.filter((entry) => entry.available && entry.task_eligible && entry.task_start_available);
        setDefinitions(eligible);
        setDefinitionId((current) => (
          eligible.some((entry) => entry.definition_id === current)
            ? current
            : eligible[0]?.definition_id || ''
        ));
      })
      .catch(() => {
        if (!active) return;
        setDefinitions([]);
        setDeepResearchDiscoveryError('Deep Research capabilities could not be loaded. Internet research is unavailable until the service recovers.');
      });
    return () => { active = false; };
  }, []);

  const refresh = useCallback(async () => {
    if (!selectedTaskId) { setTask(null); setRuns([]); setTodos([]); setItems([]); setRunIndex(-1); return; }
    const requestedTaskId = selectedTaskId;
    const [nextTask, fetchedRuns, nextTodos] = await Promise.all([
      getAgentTask(requestedTaskId, threadId),
      getAgentTaskRuns(requestedTaskId, threadId),
      getAgentTaskTodos(requestedTaskId, threadId),
    ]);
    if (taskContextRef.current !== requestedTaskId) return;
    const nextRuns = mergeActiveAgentTaskRun(nextTask, fetchedRuns);
    setTask(nextTask);
    setRuns(nextRuns);
    setTodos(nextTodos);
    setRunIndex((current) => selectAgentTaskRunIndex(
      nextTask,
      nextRuns,
      current,
      task?.active_run_id || task?.active_run?.id,
    ));
  }, [selectedTaskId, threadId]);

  const refreshTimeline = useCallback(async (taskId: string, runId: string) => {
    const value = await getAgentTaskTimeline(taskId, runId, threadId);
    if (taskContextRef.current !== taskId) return;
    setTask(value.task);
    setItems(value.items);
  }, [threadId]);

  useEffect(() => {
    setTask(null);
    setRuns([]);
    setTodos([]);
    setItems([]);
    setRunIndex(-1);
    setError('');
    sequenceRunId.current = null;
    lastSequence.current = 0;
    void refresh().catch((value) => {
      if (taskContextRef.current === selectedTaskId) setError(String(value));
    });
  }, [refresh, selectedTaskId]);
  useEffect(() => {
    if (!shouldPollAgentTask(task)) return;
    let cancelled = false;
    let timer: number | undefined;
    const poll = async () => {
      try { await refresh(); }
      catch (value) { if (!cancelled) setError(String(value)); }
      if (!cancelled) timer = window.setTimeout(poll, AGENT_TASK_POLL_INTERVAL_MS);
    };
    timer = window.setTimeout(poll, AGENT_TASK_POLL_INTERVAL_MS);
    return () => { cancelled = true; if (timer !== undefined) window.clearTimeout(timer); };
  }, [refresh, task?.status]);
  useEffect(() => {
    if (!isRunOwnedBySelectedTask(selectedTaskId, selectedRun)) { setItems([]); return; }
    let active = true;
    const taskId = selectedTaskId;
    const runId = selectedRun.id;
    void getAgentTaskTimeline(taskId, runId, threadId).then((value) => {
      if (!active || taskContextRef.current !== taskId) return;
      setTask(value.task);
      setItems(value.items);
    }).catch((value) => {
      if (active && taskContextRef.current === taskId) setError(String(value));
    });
    return () => { active = false; };
  }, [selectedRun?.id, selectedTaskId, threadId]);

  useEffect(() => {
    if (!selectedTaskId || !selectedRun || !isRunOwnedBySelectedTask(selectedTaskId, selectedRun)) return;
    const terminalStatuses = ['completed', 'failed', 'cancelled', 'expired', 'recovery_required'];
    if (!terminalStatuses.includes(String(task?.status)) && !terminalStatuses.includes(String(selectedRun.status))) return;
    void refreshTimeline(selectedTaskId, selectedRun.id).catch((value) => setError(String(value)));
  }, [refreshTimeline, selectedRun?.id, selectedRun?.status, selectedTaskId, task?.status]);

  useEffect(() => {
    if (!isRunOwnedBySelectedTask(selectedTaskId, selectedRun) || !shouldSubscribeToAgentTaskEvents(task, selectedRun)) return;
    let active = true;
    let source: EventSource | null = null;
    let reconnectTimer: number | undefined;
    const taskId = selectedTaskId;
    const runId = selectedRun.id;
    if (sequenceRunId.current !== runId) {
      sequenceRunId.current = runId;
      lastSequence.current = 0;
    }

    const close = () => {
      active = false;
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      source?.close();
      source = null;
    };
    const connect = () => {
      if (!active) return;
      const query = new URLSearchParams({
        thread_id: threadId,
        run_id: runId,
        scope: 'run',
        after_sequence: String(lastSequence.current),
      });
      source = new EventSource(`${API_BASE}/api/agent-tasks/${encodeURIComponent(taskId)}/events?${query}`);
      source.addEventListener('task_event', (event) => {
        const sequence = Number(event.lastEventId || 0);
        if (sequence > 0 && sequence <= lastSequence.current) return;
        lastSequence.current = Math.max(lastSequence.current, sequence);
        let payload: Record<string, unknown> = {};
        try { payload = JSON.parse((event as MessageEvent).data || '{}'); } catch { payload = {}; }
        const type = String(payload.type || '');
        const terminal = isTerminalAgentTaskEvent(payload);
        if (terminal) {
          void refreshTimeline(taskId, runId).catch((value) => setError(String(value)));
          close();
          void refresh().catch((value) => setError(String(value)));
          return;
        }
        if (/^(run\.|interrupt\.|approval\.|subagent\.|artifact\.)/.test(type)) void refresh();
        if (shouldRefreshAgentTaskTimeline(payload)) {
          void refreshTimeline(taskId, runId).catch((value) => setError(String(value)));
        }
      });
      source.onerror = () => {
        source?.close();
        source = null;
        if (active) reconnectTimer = window.setTimeout(connect, AGENT_SSE_RECONNECT_INTERVAL_MS);
      };
    };
    connect();
    return close;
  }, [refresh, refreshTimeline, selectedRun?.id, selectedRun?.status, selectedTaskId, task?.status, threadId]);

  const launch = async (objective: string) => {
    if (!definitionId) { setError('Select an available agent definition first.'); return; }
    if (!contextBudget) { setError('Context budget settings are still loading.'); return; }
    setBusy(true); setError('');
    try {
      const created = await createAgentTask(threadId, {
        definition_id: definitionId,
        objective,
        llm_model: model,
        context_window: contextWindow,
        context_budget: packingBudgetPayload(contextBudget),
        web_search_mode: webSearchMode,
      });
      const started = await commandAgentTask(created.id, threadId, 'start', created.version);
      onTaskSelect(started.id);
    } catch (value) { setError(value instanceof Error ? value.message : String(value)); }
    finally { setBusy(false); }
  };

  const command = async (action: 'start' | 'pause' | 'resume' | 'cancel' | 'retry') => {
    if (!task) return;
    if (action === 'cancel' && !window.confirm('Stop this Deep Research task? In-progress work on the current attempt will be cancelled.')) {
      return;
    }
    setBusy(true); setError('');
    try { setTask(await commandAgentTask(task.id, threadId, action, task.version)); await refresh(); }
    catch (value) { setError(value instanceof Error ? value.message : String(value)); await refresh(); }
    finally { setBusy(false); }
  };

  const openTrace = async () => {
    if (!selectedRun || !onOpenTrace) return;
    const run = selectedRun;
    const taskId = task?.id;
    const running = !['completed', 'failed', 'cancelled', 'expired', 'recovery_required', 'awaiting_human'].includes(run.status);
    const descriptor = {
      id: run.id,
      threadId,
      messageId: `agent-task:${taskId}:${run.id}`,
      label: `Deep Research · attempt ${run.attempt}`,
      status: run.status,
      running,
    };
    onOpenTrace(descriptor);
    const details = await getAgentRun(run.id, threadId);
    const detailsRunning = !['completed', 'failed', 'cancelled', 'expired', 'recovery_required', 'awaiting_human'].includes(details.status);
    onOpenTrace({
      ...descriptor,
      status: details.status,
      running: detailsRunning,
      runDetails: details,
    });
  };

  const decide = async (
    action: AgentRunResumeAction,
    options?: { selectedOptionIds?: string[]; editedPayload?: Record<string, unknown>; approvalScope?: 'once' | 'session' | 'always' | 'deny' },
  ) => {
    const pending = selectedRun?.pending_interrupt;
    if (!selectedRun || !pending) return;
    const responseOperation = runtimeInterruptResponseOperation(pending);
    if (!responseOperation) {
      setDecisionError('This approval request has an invalid runtime response contract.');
      return;
    }
    if (!isRuntimeOperationEnabled(effectiveSelectedRunCapabilities, responseOperation)) return;
    setDecisionSubmitting(action);
    setDecisionError('');
    try {
      await resumeAgentRun(selectedRun.id, {
        action,
        interrupt_id: pending.interrupt_id,
        resume_token: pending.resume_token,
        resume_version: pending.resume_version,
        thread_id: threadId,
        selected_option_ids: options?.selectedOptionIds,
        edited_payload: options?.editedPayload,
        client_metadata: { source: 'deep_research_task_panel' },
        approval_scope: options?.approvalScope === 'deny' ? undefined : options?.approvalScope,
      });
      await refresh();
    } catch (value) { setDecisionError(value instanceof Error ? value.message : String(value)); }
    finally { setDecisionSubmitting(null); }
  };

  const effectiveSelectedRunCapabilities = runtimeCapabilityResponseMatchesRun(
    selectedRunCapabilities,
    selectedRun?.id || '',
  ) ? selectedRunCapabilities : null;

  const taskControls = useMemo(() => TASK_CONTROL_CATALOG.map((control) => ({
    ...control,
    availability: runtimeOperationAvailability(activeTaskCapabilities, control.operation),
  })).filter((control) => control.availability.visible), [activeTaskCapabilities]);
  const theme = useTheme();
  const compactLifecycle = useMediaQuery(theme.breakpoints.down('md'));
  const [lifecycleMenuAnchor, setLifecycleMenuAnchor] = useState<HTMLElement | null>(null);
  const frozen = Boolean(selectedRun);
  const displayedContextWindow = frozen
    ? Number(task?.configuration?.context_window || contextWindow)
    : contextWindow;
  const configuredWebMode = String(task?.configuration?.web_search_mode || 'off') as 'off' | 'ask' | 'on';
  const selectedDefinition = definitions.find((entry) => entry.definition_id === definitionId);
  const selectedAgentId = frozen ? String(task?.workflow_id || definitionId) : definitionId;
  const agentSelectOptions = useMemo(() => {
    const options = definitions.map((entry) => ({
      definition_id: entry.definition_id,
      display_name: entry.display_name,
    }));
    if (selectedAgentId && !options.some((entry) => entry.definition_id === selectedAgentId)) {
      options.push({ definition_id: selectedAgentId, display_name: selectedAgentId });
    }
    return options;
  }, [definitions, selectedAgentId]);
  const agentSelectValue = agentSelectOptions.some((entry) => entry.definition_id === selectedAgentId)
    ? selectedAgentId
    : (agentSelectOptions[0]?.definition_id || '');
  const definitionFields = selectedDefinition?.configuration.fields || [];
  const modelField = definitionFields.find((field) => field.id === 'llm_model');
  const contextWindowField = definitionFields.find((field) => field.id === 'context_window');
  const webSearchField = definitionFields.find((field) => field.id === 'web_search_mode');
  const requestedWebUnavailable = webSearchMode !== 'off' && webSearchField?.enabled === false;
  const effectiveWebMode = frozen ? configuredWebMode : webSearchMode;
  const reviewGuidanceTexts = useMemo(() => {
    const context = task?.configuration?.result_review_context;
    if (!Array.isArray(context)) return [];
    return context
      .map((entry) => String((entry as { followup_input?: string }).followup_input || ''))
      .filter(Boolean);
  }, [task?.configuration?.result_review_context]);
  const objectiveWebMismatch = Boolean(
    task?.objective
    && effectiveWebMode === 'off'
    && effectiveObjectiveRequestsLiveWeb(task.objective, reviewGuidanceTexts),
  );
  const pendingInterrupt = selectedRun?.pending_interrupt?.status === 'pending' ? selectedRun.pending_interrupt : null;
  const liveBudgetSnapshot = useMemo(() => budgetUsageFromTask(task), [task]);
  const reviewTodoTitles = useMemo(() => {
    const map: Record<string, string> = {};
    todos.forEach((todo) => {
      if (todo.id && todo.title) map[todo.id] = todo.title;
    });
    return map;
  }, [todos]);
  const reviewIssueCounts = useMemo(
    () => countReviewIssues(pendingInterrupt, reviewTodoTitles),
    [pendingInterrupt, reviewTodoTitles],
  );
  const isTaskPauseInterrupt = pendingInterrupt?.type === 'task_pause' || pendingInterrupt?.node_id === 'task_pause_gate';
  const approvalTodoIds = Array.isArray(pendingInterrupt?.approval_scope?.todo_ids)
    ? pendingInterrupt.approval_scope.todo_ids.map(String)
    : [];
  const approvalScopeOptions = approvalTodoIds.map((id) => {
    const todo = todos.find((item) => item.id === id);
    return { id, label: todo?.title || id, description: todo?.description };
  });
  useEffect(() => setDecisionError(''), [pendingInterrupt?.interrupt_id]);

  const interactionDescriptors = useMemo(() => {
    const candidates: Array<{ id: 'run.send_followup' | 'run.interrupt_with_input' | 'run.steer_live'; label: string; placeholder: string }> = [
      { id: 'run.send_followup' as const, label: 'Follow up', placeholder: 'Send input after the current run finishes…' },
      { id: 'run.interrupt_with_input' as const, label: 'Interrupt with input', placeholder: 'Interrupt the run and continue with new input…' },
      { id: 'run.steer_live' as const, label: 'Steer live', placeholder: 'Guide the active run without replacing it…' },
    ];
    return candidates
      .map((item) => ({ ...item, availability: runtimeOperationAvailability(effectiveSelectedRunCapabilities, item.id) }))
      .filter((item) => item.availability.visible);
  }, [effectiveSelectedRunCapabilities]);
  const responseOperation = runtimeInterruptResponseOperation(pendingInterrupt);
  const isResultReview = pendingInterrupt?.response_operation === 'task.result_review.respond';
  const isBudgetReview = pendingInterrupt?.response_operation === 'task.budget_review.respond';
  const resultReviewAvailability = runtimeOperationAvailability(
    effectiveSelectedRunCapabilities,
    'task.result_review.respond',
  );
  const budgetReviewAvailability = runtimeOperationAvailability(effectiveSelectedRunCapabilities, 'task.budget_review.respond');
  const decisionVisible = isTaskPauseInterrupt || (isResultReview ? resultReviewAvailability.visible
    : responseOperation ? runtimeOperationAvailability(effectiveSelectedRunCapabilities, responseOperation).visible : false);
  const courseCorrectionAvailability = runtimeOperationAvailability(effectiveSelectedRunCapabilities, 'task.course_correction.submit');
  const showCourseCorrectionForm = Boolean(
    task
    && selectedRun
    && courseCorrectionAvailability.visible
    && ['queued', 'running'].includes(task.status),
  );
  const hasCourseCorrectionHistory = Boolean(task?.course_corrections?.length);
  const invalidInterruptContract = Boolean(pendingInterrupt && !responseOperation && !isResultReview && !isBudgetReview);
  const reviewPanelOpen = Boolean(pendingInterrupt && decisionVisible && (isResultReview || isBudgetReview));
  const reviewPanelHeader = pendingInterrupt ? (
    <>
      <DeepResearchBudgetSummary pending={pendingInterrupt} task={task} />
      <DeepResearchDecisionLead
        pending={pendingInterrupt}
        task={task}
        isBudgetReview={isBudgetReview}
        todoTitles={reviewTodoTitles}
      />
    </>
  ) : null;
  const timelineCitationSources = useMemo(
    () => citationSourcesFromTimelineItems(items),
    [items],
  );
  const respondToResultReview = async (decision: 'accept' | 'retry_with_input') => {
    if (!task || !selectedRun || !pendingInterrupt) return;
    const followup = decision === 'retry_with_input' ? reviewGuidance : undefined;
    if (decision === 'retry_with_input' && !followup?.trim()) return;
    setDecisionSubmitting('approve');
    setDecisionError('');
    try {
      await respondToAgentTaskResultReview(task.id, threadId, {
        run_id: selectedRun.id,
        interrupt_id: pendingInterrupt.interrupt_id,
        expected_version: task.version,
        decision,
        followup_input: followup?.trim(),
      });
      setReviewGuidance('');
      await refresh();
    } catch (value) { setDecisionError(value instanceof Error ? value.message : String(value)); }
    finally { setDecisionSubmitting(null); }
  };
  const respondToBudgetReview = async (decision: 'continue' | 'accept_partial' | 'steer') => {
    if (!task || !selectedRun || !pendingInterrupt) return;
    if (decision === 'steer' && !reviewGuidance.trim()) return;
    setDecisionSubmitting('approve'); setDecisionError('');
    try {
      await respondToAgentTaskBudgetReview(task.id, threadId, {
        run_id: selectedRun.id, interrupt_id: pendingInterrupt.interrupt_id,
        expected_version: task.version, decision,
        guidance: decision === 'steer' ? reviewGuidance.trim() : undefined,
      });
      setReviewGuidance(''); await refresh();
    } catch (value) { setDecisionError(value instanceof Error ? value.message : String(value)); }
    finally { setDecisionSubmitting(null); }
  };
  useEffect(() => {
    if (interactionDescriptors.length && !interactionDescriptors.some((operation) => operation.id === interactionOperation)) {
      setInteractionOperation(interactionDescriptors[0].id);
    }
  }, [interactionDescriptors, interactionOperation]);

  return <ConversationPanelTemplate
    ref={panelRef}
    sx={{ p: 1 }}
    transcriptSx={reviewPanelOpen ? {
      flex: '0 0 auto',
      flexShrink: 0,
      minHeight: 96,
      maxHeight: '22vh',
      overflow: 'hidden',
    } : {
      flex: 1,
      minHeight: 0,
    }}
    decisionSx={reviewPanelOpen ? { flex: '1 1 0', minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' } : undefined}
    header={<ConversationHeader
      models={models}
      model={frozen ? String(task?.configuration?.llm_model || model) : model}
      contextWindow={displayedContextWindow}
      contextBudget={contextBudget}
      defaultContextBudget={defaultContextBudget}
      disabled={frozen || modelField?.read_only === true}
      contextWindowDisabled={frozen || contextWindowField?.read_only === true}
      contextBudgetDisabled={contextBudgetDisabled}
      onModelChange={onModelChange}
      onContextWindowChange={onContextWindowChange}
      onContextBudgetChange={onContextBudgetChange}
      leading={<><Tooltip title="Back to chat"><IconButton size="small" onClick={onBack}><ArrowBackIcon fontSize="small" /></IconButton></Tooltip>{embeddingControl}</>}
      beforeModelControls={webSearchField ? renderWebControl(
        frozen ? configuredWebMode : webSearchMode,
        frozen || webSearchField.enabled === false,
      ) : null}
      afterModelControls={agentSelectValue ? (
        <WorkbenchSelect
          label="Select agent"
          aria-label="Agent definition"
          value={agentSelectValue}
          disabled={frozen}
          onChange={setDefinitionId}
        >
          {agentSelectOptions.map((entry) => (
            <MenuItem key={entry.definition_id} value={entry.definition_id}>{entry.display_name}</MenuItem>
          ))}
        </WorkbenchSelect>
      ) : null}
      trailingActions={<DeepResearchTaskPicker threadId={threadId} selectedTaskId={selectedTaskId} onSelect={onTaskSelect} />}
    />}
    status={<>
      {error && <Alert severity="error" sx={{ mb: 1 }}>{error}</Alert>}
      {deepResearchDiscoveryError && <Alert severity="warning" sx={{ mb: 1 }}>{deepResearchDiscoveryError}</Alert>}
      {requestedWebUnavailable && <Alert severity="warning" sx={{ mb: 1 }}>The selected definition does not allow web search.</Alert>}
      {objectiveWebMismatch && <Alert severity="warning" sx={{ mb: 1 }}>
        This objective asks for web benchmarks or external sources, but web search is off for this task. Industry comparisons will be missing unless you start a new task with web search enabled.
      </Alert>}
      {runtimeControlError && <Alert severity="warning" sx={{ mb: 1 }}>{runtimeControlError}</Alert>}
      {task?.status === 'recovery_required' && <Alert severity="warning" sx={{ mb: 1 }}>
        Runtime execution finished, but its product-state update could not be applied safely. Retry the task or ask an administrator to reconcile this run.
      </Alert>}
      {frozen && task && <Alert severity="info" sx={{ mb: 1 }} icon={false}>
        Model, agent, and context window are locked for this task. Start a new Deep Research task to change them.
      </Alert>}
      {liveBudgetSnapshot && task && !pendingInterrupt && <Box sx={{ mb: 1 }}>
        <DeepResearchBudgetChips snapshot={liveBudgetSnapshot} />
      </Box>}
      {task?.research_canvases?.length ? (
        <DeepResearchTaskDeliverables
          canvases={task.research_canvases}
          selectedRunId={selectedRun?.id}
          onOpenCanvas={onOpenCanvas}
        />
      ) : null}
    </>}
    transcript={<ConversationTranscriptFrame sx={reviewPanelOpen ? { flex: '0 0 auto', overflow: 'hidden', maxHeight: '100%' } : undefined}>{items.map((item) => <TimelineBubble
      key={item.id}
      item={item}
      taskId={task?.id || ''}
      threadId={threadId}
      onSaveToMemory={onSaveToMemory}
      onOpenCanvas={onOpenCanvas}
      citationSources={timelineCitationSources}
      copied={copiedId === item.id}
      active={activeItemId === item.id}
      onCopy={() => copyItem(item)}
      onReadAloud={() => readItem(item.id)}
      rootRef={(node) => { if (node) itemRefs.current.set(item.id, node); else itemRefs.current.delete(item.id); }}
      onSelectAttempt={(attempt) => {
        const index = runs.findIndex((run) => run.attempt === attempt);
        if (index >= 0) setRunIndex(index);
      }}
    />)}</ConversationTranscriptFrame>}
    decision={invalidInterruptContract ? <Alert severity="error" sx={{ m: 2 }}>This human-input request has an invalid runtime response contract.</Alert> : !decisionVisible ? undefined : pendingInterrupt && isTaskPauseInterrupt ? <Box sx={{ p: 2 }}>
      <Typography variant="subtitle2">Deep research paused</Typography>
      <Typography variant="body2" color="text.secondary">The task is paused at a durable checkpoint. Use Resume to continue or Cancel to stop the task.</Typography>
      {decisionError ? <Alert severity="error" sx={{ mt: 1 }}>{decisionError}</Alert> : null}
    </Box> : pendingInterrupt && isBudgetReview ? <ResizableDecisionPanel
      title={pendingInterrupt.title || 'Research budget reached'}
      variant="approval"
      rootRef={panelRef}
      horizontalInset={1}
      fillColumn
      fixedHeader={reviewPanelHeader}
      footer={<DeepResearchDecisionStickyFooter>
        <DeepResearchGuidanceField
          label="Optional guidance for the next tranche"
          helperText="Continue and steer start another research tranche and use more budget."
          value={reviewGuidance}
          onChange={setReviewGuidance}
        />
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
          <Button size="small" variant="contained" disabled={Boolean(decisionSubmitting) || !pendingInterrupt.provisional_answer || !budgetReviewAvailability.enabled} onClick={() => void respondToBudgetReview('accept_partial')}>Finish with partial answer</Button>
          <Button size="small" variant="outlined" disabled={Boolean(decisionSubmitting) || !budgetReviewAvailability.enabled} title="Uses another research tranche" onClick={() => void respondToBudgetReview('continue')}>Continue research (more budget)</Button>
          <Button size="small" variant="outlined" disabled={Boolean(decisionSubmitting) || !reviewGuidance.trim() || !budgetReviewAvailability.enabled} title="Uses another research tranche" onClick={() => void respondToBudgetReview('steer')}>Apply guidance and continue</Button>
        </Box>
        {decisionError ? <Alert severity="error">{decisionError}</Alert> : null}
      </DeepResearchDecisionStickyFooter>}
    >
      <DeepResearchIssueList pending={pendingInterrupt} todoTitles={reviewTodoTitles} defaultExpanded />
      <DeepResearchProvisionalAnswer
        content={pendingInterrupt.provisional_answer}
        emptyMessage="No usable provisional answer was produced. Continue or steer the research."
        citationSources={timelineCitationSources}
      />
    </ResizableDecisionPanel> : pendingInterrupt && isResultReview ? <ResizableDecisionPanel
      title={pendingInterrupt.title || 'Review incomplete result'}
      variant="approval"
      rootRef={panelRef}
      horizontalInset={1}
      fillColumn
      fixedHeader={reviewPanelHeader}
      footer={<DeepResearchDecisionStickyFooter>
        <DeepResearchGuidanceField
          label="Guidance for the next research tranche"
          helperText="Continuing with guidance resets the budget tranche counters and may start a linked attempt."
          value={reviewGuidance}
          onChange={setReviewGuidance}
          required
        />
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
          <Button
            size="small"
            variant="contained"
            disabled={Boolean(decisionSubmitting) || !resultReviewAvailability.enabled}
            title={resultReviewAvailability.disabledReason}
            onClick={() => void respondToResultReview('accept')}
          >
            {reviewIssueCounts.limitationCount > 0
              ? `Finish with this result (${reviewIssueCounts.limitationCount} limitation${reviewIssueCounts.limitationCount === 1 ? '' : 's'})`
              : 'Finish with this result'}
          </Button>
          <Button
            size="small"
            variant="outlined"
            disabled={Boolean(decisionSubmitting) || !reviewGuidance.trim() || !resultReviewAvailability.enabled}
            title={resultReviewAvailability.disabledReason}
            onClick={() => void respondToResultReview('retry_with_input')}
          >
            {resultReviewContinueLabel(reviewIssueCounts.limitationCount)}
          </Button>
        </Box>
        {decisionError ? <Alert severity="error">{decisionError}</Alert> : null}
      </DeepResearchDecisionStickyFooter>}
    >
      <DeepResearchIssueList pending={pendingInterrupt} todoTitles={reviewTodoTitles} defaultExpanded />
      {pendingInterrupt.provisional_answer ? (
        <DeepResearchProvisionalAnswer
          content={pendingInterrupt.provisional_answer}
          citationSources={timelineCitationSources}
          defaultCollapsed={false}
        />
      ) : null}
    </ResizableDecisionPanel> : pendingInterrupt && responseOperation ? <HumanReviewDecisionPanel
      interrupt={pendingInterrupt}
      submitting={decisionSubmitting}
      error={decisionError || null}
      disabled={!runtimeOperationAvailability(effectiveSelectedRunCapabilities, responseOperation).visible || !isRuntimeOperationEnabled(effectiveSelectedRunCapabilities, responseOperation)}
      disabledReason={runtimeOperationAvailability(effectiveSelectedRunCapabilities, responseOperation).disabledReason}
      scopeOptions={approvalScopeOptions}
      rootRef={panelRef}
      onAction={(action, options) => void decide(action, options)}
    /> : undefined}
    composer={<Box sx={{ pb: 1 }}>
      {task && <Box sx={{ borderTop: 1, borderColor: 'divider', py: 0.75, px: 1, mb: 1 }}>
        <Stack direction="row" alignItems="center" spacing={0.75} flexWrap="wrap">
          <Chip size="small" label={formatAgentTaskStatus(task.status)} color={task.status === 'completed' ? 'success' : task.status === 'failed' ? 'error' : task.status === 'recovery_required' ? 'warning' : 'primary'} />
          <Typography variant="caption">Attempt {selectedRun?.attempt || 0} of {runs.length}</Typography>
          <IconButton size="small" aria-label="Previous attempt" disabled={runIndex <= 0} onClick={() => setRunIndex((value) => value - 1)}><NavigateBeforeIcon fontSize="small" /></IconButton>
          <IconButton size="small" aria-label="Next attempt" disabled={runIndex < 0 || runIndex >= runs.length - 1} onClick={() => setRunIndex((value) => value + 1)}><NavigateNextIcon fontSize="small" /></IconButton>
          <Box sx={{ flex: 1 }} />
          {compactLifecycle ? <>
            <Button
              size="small"
              startIcon={<MoreVertIcon />}
              onClick={(event) => setLifecycleMenuAnchor(event.currentTarget)}
            >
              Actions
            </Button>
            <Menu anchorEl={lifecycleMenuAnchor} open={Boolean(lifecycleMenuAnchor)} onClose={() => setLifecycleMenuAnchor(null)}>
              {taskControls.map(({ action, label, availability }) => (
                <MenuItem
                  key={action}
                  disabled={busy || !availability.enabled}
                  title={availability.disabledReason}
                  onClick={() => { setLifecycleMenuAnchor(null); void command(action); }}
                >
                  {label}
                </MenuItem>
              ))}
              <MenuItem disabled={!selectedRun || !onOpenTrace} onClick={() => { setLifecycleMenuAnchor(null); void openTrace(); }}>Debug Trace</MenuItem>
            </Menu>
          </> : <>
            {taskControls.map(({ action, label, availability }) => (
              <Button
                key={action}
                size="small"
                color={action === 'cancel' ? 'error' : 'primary'}
                disabled={busy || !availability.enabled}
                title={availability.disabledReason}
                onClick={() => void command(action)}
              >{label}</Button>
            ))}
            <Button size="small" startIcon={<BugReportIcon />} disabled={!selectedRun || !onOpenTrace} onClick={() => void openTrace()}>Debug Trace</Button>
          </>}
        </Stack>
        <LinearProgress variant="determinate" value={task.progress} sx={{ mt: 0.75 }} />
      </Box>}
      {!task ? (
        <ConversationComposer placeholder="Describe a new Deep Research objective…" busy={busy} disabled={!model || requestedWebUnavailable} onSubmit={(value) => void launch(value)} />
      ) : interactionDescriptors.length > 0 || showCourseCorrectionForm || hasCourseCorrectionHistory ? <>
      {showCourseCorrectionForm ? <Stack direction="row" spacing={1} sx={{ mb: 1 }} alignItems="flex-start">
        <TextField fullWidth multiline minRows={2} label="Redirect research after active workers finish" value={courseCorrection} onChange={(event) => setCourseCorrection(event.target.value)} />
        <Button variant="outlined" disabled={!courseCorrection.trim() || !courseCorrectionAvailability.enabled || !selectedRun} onClick={() => {
          if (!selectedRun || !task) return;
          void submitAgentTaskCourseCorrection(task.id, threadId, { run_id: selectedRun.id, expected_version: task.version, instruction: courseCorrection.trim() })
            .then((response) => {
              setCourseCorrection('');
              setCourseCorrectionStatus(
                response.delivery_state === 'linked'
                  ? 'Correction linked to a follow-up run.'
                  : response.delivery_state === 'incorporated'
                    ? 'Correction is incorporated into this attempt and remains active until the result verifies coverage.'
                  : response.delivery_state === 'satisfied'
                    ? 'Correction was verified in the final result.'
                  : response.delivery_state === 'unresolved'
                    ? 'Execution finished without fully resolving this correction. Review the partial result or retry.'
                  : response.delivery_state === 'delivered'
                    ? 'Correction delivered and waiting for the next safe planning boundary.'
                    : 'Correction accepted and queued for delivery.',
              );
              return refresh();
            })
            .catch((reason) => setError(reason instanceof Error ? reason.message : String(reason)));
        }}>Redirect research</Button>
      </Stack> : null}
      {courseCorrectionStatus ? <Alert severity="info" sx={{ mb: 1 }}>{courseCorrectionStatus}</Alert> : null}
      {task.course_corrections?.length ? <Stack spacing={0.75} sx={{ mb: 1 }}>
        {task.course_corrections.map((correction) => <Alert
          key={correction.correction_id}
          severity={correction.delivery_state === 'unresolved' ? 'warning' : correction.delivery_state === 'satisfied' ? 'success' : 'info'}
          icon={false}
        >
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
            <Chip size="small" label={correction.delivery_state.replaceAll('_', ' ')} />
            <Typography variant="body2">{correction.instruction}</Typography>
          </Stack>
          {correction.delivery_state === 'unresolved' ? <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
            {correction.runtime_outcome?.unresolved_reason || 'Execution finished without fully addressing this redirect. Retry or accept the partial result.'}
          </Typography> : null}
        </Alert>)}
      </Stack> : null}
      <Stack direction="row" spacing={1} sx={{ mb: 1 }}>
        {interactionDescriptors.map((operation) => <Button
          key={operation.id}
          size="small"
          variant={interactionOperation === operation.id ? 'contained' : 'outlined'}
          disabled={!operation.availability.enabled}
          title={operation.availability.disabledReason}
          onClick={() => setInteractionOperation(operation.id)}
        >{operation.label}</Button>)}
      </Stack>
      {interactionDescriptors.length > 0 && <ConversationComposer
          placeholder={interactionDescriptors.find((operation) => operation.id === interactionOperation)?.placeholder || 'Send runtime input…'}
          busy={busy}
          disabled={!interactionDescriptors.find((operation) => operation.id === interactionOperation)?.availability.enabled}
          onSubmit={async (value) => {
            if (!selectedRun) return;
            setBusy(true); setError('');
            try {
              if (interactionOperation === 'run.send_followup') await sendAgentRunFollowup(selectedRun.id, threadId, value);
              else if (interactionOperation === 'run.interrupt_with_input') await interruptAgentRunWithInput(selectedRun.id, threadId, value);
              else await steerAgentRunLive(selectedRun.id, threadId, value);
              await refresh();
            } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
            finally { setBusy(false); }
          }}
        />}
      </> : (
        <Box sx={{ px: 2, py: 1 }}>
          <Typography variant="body2" color="text.secondary">
            {task.status === 'cancelled'
              ? 'This task was cancelled. Start a new Deep Research task for a new objective.'
              : task.status === 'running' || task.status === 'queued'
                ? (taskControls.some((control) => control.action === 'pause' && control.availability.enabled)
                  ? 'Research is running. Use Actions above to pause or cancel.'
                  : 'Research is running. Use Actions above to cancel if needed.')
                : task.status === 'awaiting_approval'
                  ? 'Review the request above to continue or finish.'
                  : task.status === 'paused'
                    ? 'Research is paused. Resume or cancel from Actions above.'
                    : task.status === 'recovery_required'
                      ? 'Runtime execution stopped at a recovery boundary. Retry or cancel from Actions above.'
                      : task.status === 'completed'
                        ? 'This run is complete. Select New Deep Research task for a follow-up objective.'
                        : taskControls.length
                          ? 'Use Actions above for the next step.'
                          : 'No further actions are available for this task.'}
          </Typography>
        </Box>
      )}
    </Box>}
  />;
}
