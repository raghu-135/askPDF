import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert, Box, Button, Chip,
  Divider, IconButton, LinearProgress, ListItemIcon, ListItemText,
  Menu, MenuItem, Stack, Tooltip, Typography,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import DeleteIcon from '@mui/icons-material/Delete';
import NavigateBeforeIcon from '@mui/icons-material/NavigateBefore';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import PsychologyIcon from '@mui/icons-material/Psychology';
import TravelExploreIcon from '@mui/icons-material/TravelExplore';
import {
  API_BASE,
  agentRunEventsUrl,
  commandAgentTask,
  createAgentTask,
  deleteAgentTask,
  downloadAgentTaskArtifact,
  listAgentDefinitions,
  getAgentRun,
  getAgentRunCapabilities,
  getAgentTask,
  getAgentTaskRuns,
  getAgentTaskTodos,
  getAgentTaskTimeline,
  listAgentTasks,
  resumeAgentRun,
  sendAgentRunFollowup,
  interruptAgentRunWithInput,
  steerAgentRunLive,
  updateAgentRunState,
  type AgentTaskRun,
  type AgentTaskSummary,
  type AgentTaskTimelineItem,
  type AgentTaskTodo,
  type AgentRunResumeAction,
  type AgentRunDetails,
  type AgentRuntimeCapabilityResponse,
  type AgentDefinitionCatalogEntry,
  type BuilderTestStreamEnvelope,
} from '../lib/api';
import {
  isRunOwnedBySelectedTask,
  isTerminalAgentTaskEvent,
  mergeActiveAgentTaskRun,
  shouldPollAgentTask,
  shouldRefreshAgentTaskTimeline,
  shouldSubscribeToAgentTaskEvents,
} from '../lib/deep-research-ui-state';
import { isCurrentRuntimeCapabilityRequest, isRuntimeOperationEnabled, runtimeCapabilityResponseMatchesRun, runtimeInterruptResponseOperation, runtimeOperationAvailability, TASK_CONTROL_CATALOG } from '../lib/runtime-capabilities';
import { withRetry } from '../lib/retry-utils';
import {
  deriveConversationSentences,
  type ConversationSentence,
  type ConversationSentenceCache,
} from '../lib/chat-sentence-cache';
import type { ChatTraceDescriptor } from './ChatInterface';
import { buildLiveTraceView } from './agent-debug/agent-trace-projection';
import {
  ConversationComposer,
  ConversationArtifactList,
  ConversationDisclosure,
  ConversationHeader,
  ConversationMessageActions,
  HumanReviewDecisionPanel,
  ConversationMessageBubble,
  ConversationPanelTemplate,
  ConversationTranscriptFrame,
  SourceList,
} from './conversation';

function requiredPositiveMilliseconds(name: string, raw: string | undefined): number {
  const value = Number(raw);
  if (!raw || !Number.isFinite(value) || value <= 0) {
    throw new Error(`Required environment variable ${name} must be a positive number`);
  }
  return value;
}

const AGENT_TASK_POLL_INTERVAL_MS = requiredPositiveMilliseconds(
  'NEXT_PUBLIC_AGENT_TASK_POLL_INTERVAL_MS',
  process.env.NEXT_PUBLIC_AGENT_TASK_POLL_INTERVAL_MS,
);
const AGENT_SSE_RECONNECT_INTERVAL_MS = requiredPositiveMilliseconds(
  'NEXT_PUBLIC_AGENT_SSE_RECONNECT_INTERVAL_MS',
  process.env.NEXT_PUBLIC_AGENT_SSE_RECONNECT_INTERVAL_MS,
);


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

  return <>
    <Button
      size="small"
      color="inherit"
      startIcon={<TravelExploreIcon fontSize="small" />}
      onClick={(event) => { setAnchor(event.currentTarget); void refresh(); }}
      sx={{ maxWidth: 230, textTransform: 'none' }}
    >
      <Typography variant="body2" noWrap>{selected?.objective || 'Deep Research'}</Typography>
    </Button>
    <Menu anchorEl={anchor} open={Boolean(anchor)} onClose={() => setAnchor(null)} slotProps={{ paper: { sx: { width: 360, maxWidth: '90vw' } } }}>
      <MenuItem onClick={() => { onSelect(null); setAnchor(null); }}>
        <ListItemIcon><TravelExploreIcon fontSize="small" /></ListItemIcon>
        <ListItemText primary="New Deep Research task" secondary="Start a separate long-running research run" />
      </MenuItem>
      <Divider />
      {tasks.map((task) => <MenuItem key={task.id} selected={task.id === selectedTaskId} onClick={() => { onSelect(task.id); setAnchor(null); }}>
        <ListItemText
          primary={task.objective}
          secondary={`${task.status.replaceAll('_', ' ')} · attempt ${task.run_attempt || 0}`}
          slotProps={{ primary: { noWrap: true }, secondary: { noWrap: true } }}
        />
        {['completed', 'failed', 'expired', 'cancelled'].includes(task.status) && <IconButton size="small" color="error" disabled={busy} onClick={(event) => void remove(event, task)} aria-label="Delete task">
          <DeleteIcon fontSize="small" />
        </IconButton>}
      </MenuItem>)}
      {!tasks.length && <MenuItem disabled>No research tasks yet</MenuItem>}
    </Menu>
  </>;
}


function TimelineBubble({
  item,
  taskId,
  threadId,
  onSaveToMemory,
  copied,
  active,
  onCopy,
  onReadAloud,
  rootRef,
  onSelectAttempt,
}: {
  item: AgentTaskTimelineItem;
  taskId: string;
  threadId: string;
  onSaveToMemory?: (content: string) => void;
  copied: boolean;
  active: boolean;
  onCopy: () => void;
  onReadAloud: () => void;
  rootRef: React.Ref<HTMLLIElement>;
  onSelectAttempt?: (attempt: number) => void;
}) {
  const isObjective = item.type === 'objective';
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
  return <ConversationMessageBubble
    rootRef={rootRef}
    role={isObjective ? 'user' : 'assistant'}
    content={item.primary_content}
    active={active}
    wide={!isObjective}
    badge={<Chip size="small" label={`${item.type.replaceAll('_', ' ')} · ${item.status}`} sx={{ mb: 1 }} color={['todo_failure', 'run_failure'].includes(item.type) ? 'error' : item.type === 'final_report' ? 'success' : 'default'} />}
    actions={<ConversationMessageActions copied={copied} readActive={active} onCopy={onCopy} onReadAloud={onReadAloud}>
      {onSaveToMemory && ['todo_result', 'final_report'].includes(item.type) && <Button size="small" onClick={() => onSaveToMemory(item.primary_content)}>Save to memory</Button>}
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
  selectedTaskId,
  embeddingControl,
  renderWebControl,
  webSearchMode,
  onModelChange,
  onContextWindowChange,
  onTaskSelect,
  onBack,
  onOpenTrace,
  onSaveToMemory,
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
  selectedTaskId: string | null;
  embeddingControl: React.ReactNode;
  renderWebControl: (mode: 'off' | 'ask' | 'on', disabled: boolean) => React.ReactNode;
  webSearchMode: 'off' | 'ask' | 'on';
  onModelChange: (model: string) => void;
  onContextWindowChange: (value: number) => void;
  onTaskSelect: (taskId: string | null) => void;
  onBack: () => void;
  onOpenTrace?: (trace: ChatTraceDescriptor) => void;
  onSaveToMemory?: (content: string) => void;
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
  const [decisionError, setDecisionError] = useState('');
  const [error, setError] = useState('');
  const [definitions, setDefinitions] = useState<AgentDefinitionCatalogEntry[]>([]);
  const [definitionId, setDefinitionId] = useState('');
  const [deepResearchDiscoveryError, setDeepResearchDiscoveryError] = useState('');
  const [runtimeControlError, setRuntimeControlError] = useState('');
  const [liveTraceEvents, setLiveTraceEvents] = useState<BuilderTestStreamEnvelope[]>([]);
  const [traceLiveRequested, setTraceLiveRequested] = useState(false);
  const [selectedRunCapabilities, setSelectedRunCapabilities] = useState<AgentRuntimeCapabilityResponse | null>(null);
  const [activeTaskCapabilities, setActiveTaskCapabilities] = useState<AgentRuntimeCapabilityResponse | null>(null);
  const [interactionOperation, setInteractionOperation] = useState<'run.send_followup' | 'run.interrupt_with_input' | 'run.steer_live'>('run.send_followup');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const lastSequence = useRef(0);
  const sequenceRunId = useRef<string | null>(null);
  const liveTraceEventsRef = useRef<BuilderTestStreamEnvelope[]>([]);
  const liveTraceRunDetailsRef = useRef<AgentRunDetails | undefined>(undefined);
  const capabilityRequestId = useRef(0);
  const activeCapabilityRequestId = useRef(0);
  const taskContextRef = useRef(selectedTaskId);
  taskContextRef.current = selectedTaskId;
  const sentenceCacheRef = useRef<ConversationSentenceCache>(new Map());
  const itemRefs = useRef(new Map<string, HTMLLIElement>());
  const selectedRun = runs[runIndex] || null;

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
    setRunIndex((current) => current >= 0 && current < nextRuns.length ? current : nextRuns.length - 1);
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
    setTraceLiveRequested(false);
    liveTraceRunDetailsRef.current = undefined;
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
    if (!traceLiveRequested || !selectedRun || !isRunOwnedBySelectedTask(selectedTaskId, selectedRun) || !shouldSubscribeToAgentTaskEvents(task, selectedRun)) {
      liveTraceEventsRef.current = [];
      setLiveTraceEvents([]);
      return undefined;
    }
    let active = true;
    let source: EventSource | null = null;
    const runId = selectedRun.id;
    let afterSequence = 0;
    const connect = () => {
      if (!active) return;
      source = new EventSource(agentRunEventsUrl(runId, threadId, afterSequence));
      source.addEventListener('run_event', (event) => {
        let value: Record<string, any>;
        try { value = JSON.parse((event as MessageEvent).data || '{}'); } catch { return; }
        const sequence = Number(value.sequence || 0);
        if (sequence > 0 && sequence <= afterSequence) return;
        afterSequence = Math.max(afterSequence, sequence);
        const kind = String(value.kind || 'runtime.event');
        const payload = value.payload && typeof value.payload === 'object' ? value.payload : {};
        const data = {
          ...payload,
          event_id: value.event_id ?? (payload as Record<string, any>).event_id,
          sequence: value.sequence,
          attempt: value.attempt,
          occurred_at: value.occurred_at,
          parallel_groups: value.parallel_groups,
        };
        const envelope = { id: value.id || sequence, event: kind, data } as BuilderTestStreamEnvelope;
        if (active) {
          liveTraceEventsRef.current = [...liveTraceEventsRef.current, envelope];
          setLiveTraceEvents(liveTraceEventsRef.current);
          const liveTraceView = buildLiveTraceView(liveTraceEventsRef.current);
          onOpenTrace?.({
            id: runId,
            threadId,
            messageId: `agent-task:${selectedTaskId}:${runId}`,
            label: `Deep Research · attempt ${selectedRun.attempt}`,
            status: ['run.completed', 'run.failed', 'run.cancelled'].includes(kind) ? kind.slice(4) : 'running',
            liveTraceView,
            runDetails: liveTraceRunDetailsRef.current,
            running: !['run.completed', 'run.failed', 'run.cancelled'].includes(kind),
          });
        }
        if (['run.completed', 'run.failed', 'run.cancelled'].includes(kind)) {
          active = false;
          // The terminal event is the end of the live projection. Stop the
          // subscription immediately; task polling will provide the retained
          // authoritative trace without reopening the SSE stream from zero.
          setTraceLiveRequested(false);
          source?.close();
          source = null;
        }
      });
      source.onerror = () => {
        source?.close();
        source = null;
        if (active) window.setTimeout(connect, AGENT_SSE_RECONNECT_INTERVAL_MS);
      };
    };
    connect();
    return () => {
      active = false;
      source?.close();
      source = null;
    };
  }, [onOpenTrace, selectedRun?.attempt, selectedRun?.id, selectedTaskId, task?.status, threadId, traceLiveRequested]);

  useEffect(() => {
    let active = true;
    const requestId = capabilityRequestId.current + 1;
    capabilityRequestId.current = requestId;
    setSelectedRunCapabilities(null);
    setRuntimeControlError('');
    if (!selectedRun) {
      return () => { active = false; };
    }
    void withRetry(() => getAgentRunCapabilities(selectedRun.id, threadId), { maxRetries: 3, baseDelay: 500 })
      .then((result) => {
        if (!active || !isCurrentRuntimeCapabilityRequest(requestId, capabilityRequestId.current)) return;
        if (result.success) {
          if (!runtimeCapabilityResponseMatchesRun(result.data, selectedRun.id)) {
            setSelectedRunCapabilities(null);
            setRuntimeControlError('Run controls are temporarily unavailable. Refresh the run to retry.');
            return;
          }
          setSelectedRunCapabilities(result.data || null);
          setRuntimeControlError(
            result.data?.runtime_available
              ? ''
              : 'The runtime deployment is unavailable. Run controls will remain disabled until it recovers.',
          );
        } else {
          setSelectedRunCapabilities(null);
          setRuntimeControlError('Run controls are temporarily unavailable. Refresh the run to retry.');
        }
      });
    return () => { active = false; };
  }, [
    selectedRun?.id,
    selectedRun?.status,
    selectedRun?.runtime_binding_status,
    selectedRun?.pending_interrupt?.interrupt_id,
    selectedRun?.pending_interrupt?.status,
    selectedRun?.pending_interrupt?.resume_version,
    selectedRun?.pending_interrupt?.response_operation,
    task?.version,
    threadId,
  ]);

  useEffect(() => {
    let active = true;
    const requestId = activeCapabilityRequestId.current + 1;
    activeCapabilityRequestId.current = requestId;
    setActiveTaskCapabilities(null);
    const activeRunId = task?.active_run_id;
    if (!activeRunId) {
      return () => { active = false; };
    }
    void withRetry(() => getAgentRunCapabilities(activeRunId, threadId), { maxRetries: 3, baseDelay: 500 })
      .then((result) => {
        if (!active || !isCurrentRuntimeCapabilityRequest(requestId, activeCapabilityRequestId.current)) return;
        if (result.success && runtimeCapabilityResponseMatchesRun(result.data, activeRunId)) {
          setActiveTaskCapabilities(result.data || null);
          if (!result.data?.runtime_available) {
            setRuntimeControlError('The runtime deployment is unavailable. Run controls will remain disabled until it recovers.');
          }
        } else {
          setActiveTaskCapabilities(null);
        }
      });
    return () => { active = false; };
  }, [task?.active_run_id, task?.status, task?.version, threadId]);

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
    setBusy(true); setError('');
    try {
      const created = await createAgentTask(threadId, { definition_id: definitionId, objective, llm_model: model, context_window: contextWindow, web_search_mode: webSearchMode });
      const started = await commandAgentTask(created.id, threadId, 'start', created.version);
      onTaskSelect(started.id);
    } catch (value) { setError(value instanceof Error ? value.message : String(value)); }
    finally { setBusy(false); }
  };

  const command = async (action: 'start' | 'pause' | 'resume' | 'cancel' | 'retry') => {
    if (!task) return;
    setBusy(true); setError('');
    try { setTask(await commandAgentTask(task.id, threadId, action, task.version)); await refresh(); }
    catch (value) { setError(value instanceof Error ? value.message : String(value)); await refresh(); }
    finally { setBusy(false); }
  };

  const openTrace = async () => {
    if (!selectedRun || !onOpenTrace) return;
    setTraceLiveRequested(true);
    const details = await getAgentRun(selectedRun.id, threadId);
    liveTraceRunDetailsRef.current = details;
    onOpenTrace({ id: selectedRun.id, threadId, messageId: `agent-task:${task?.id}:${selectedRun.id}`, label: `Deep Research · attempt ${selectedRun.attempt}`, status: selectedRun.status, runDetails: details, liveTraceView: liveTraceEvents.length ? buildLiveTraceView(liveTraceEvents) : undefined, running: !['completed', 'failed', 'cancelled', 'expired'].includes(selectedRun.status) });
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
  const frozen = Boolean(selectedRun);
  const displayedContextWindow = frozen
    ? Number(task?.configuration?.context_window || contextWindow)
    : contextWindow;
  const configuredWebMode = String(task?.configuration?.web_search_mode || 'off') as 'off' | 'ask' | 'on';
  const frozenWebMode = task?.web_access === 'allowed_for_task'
    ? 'on'
    : task?.web_access === 'denied_for_task'
      ? 'off'
      : configuredWebMode;
  const selectedDefinition = definitions.find((entry) => entry.definition_id === definitionId);
  const definitionFields = selectedDefinition?.configuration.fields || [];
  const modelField = definitionFields.find((field) => field.id === 'llm_model');
  const contextWindowField = definitionFields.find((field) => field.id === 'context_window');
  const webSearchField = definitionFields.find((field) => field.id === 'web_search_mode');
  const requestedWebUnavailable = webSearchMode !== 'off' && webSearchField?.enabled === false;
  const pendingInterrupt = selectedRun?.pending_interrupt?.status === 'pending' ? selectedRun.pending_interrupt : null;
  const isApprovalInterrupt = pendingInterrupt?.kind === 'approval';
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
  const stateUpdateAvailability = runtimeOperationAvailability(effectiveSelectedRunCapabilities, 'run.update_state');
  const responseOperation = runtimeInterruptResponseOperation(pendingInterrupt);
  const invalidInterruptContract = Boolean(pendingInterrupt && !responseOperation);
  useEffect(() => {
    if (interactionDescriptors.length && !interactionDescriptors.some((operation) => operation.id === interactionOperation)) {
      setInteractionOperation(interactionDescriptors[0].id);
    }
  }, [interactionDescriptors, interactionOperation]);

  return <ConversationPanelTemplate
    sx={{ p: 1 }}
    header={<ConversationHeader
      models={models}
      model={frozen ? String(task?.configuration?.llm_model || model) : model}
      contextWindow={displayedContextWindow}
      disabled={frozen || modelField?.read_only === true}
      contextWindowDisabled={frozen || contextWindowField?.read_only === true}
      onModelChange={onModelChange}
      onContextWindowChange={onContextWindowChange}
      leading={<><Tooltip title="Back to chat"><IconButton size="small" onClick={onBack}><ArrowBackIcon fontSize="small" /></IconButton></Tooltip>{embeddingControl}</>}
      beforeModelControls={<Stack direction="row" spacing={1} alignItems="center">
        <Chip size="small" label={frozen ? String(task?.workflow_id || definitionId) : (selectedDefinition?.display_name || 'Select definition')} />
        {!frozen && definitions.length > 1 ? (
          <select value={definitionId} onChange={(event) => setDefinitionId(event.target.value)} aria-label="Agent definition">
            {definitions.map((entry) => <option key={entry.definition_id} value={entry.definition_id}>{entry.display_name}</option>)}
          </select>
        ) : null}
        {webSearchField ? renderWebControl(
          frozen ? frozenWebMode : webSearchMode,
          frozen || webSearchField.enabled === false,
        ) : null}
      </Stack>}
      trailingActions={<DeepResearchTaskPicker threadId={threadId} selectedTaskId={selectedTaskId} onSelect={onTaskSelect} />}
    />}
    status={<>
      {error && <Alert severity="error" sx={{ mb: 1 }}>{error}</Alert>}
      {deepResearchDiscoveryError && <Alert severity="warning" sx={{ mb: 1 }}>{deepResearchDiscoveryError}</Alert>}
      {requestedWebUnavailable && <Alert severity="warning" sx={{ mb: 1 }}>The selected definition does not allow web search.</Alert>}
      {runtimeControlError && <Alert severity="warning" sx={{ mb: 1 }}>{runtimeControlError}</Alert>}
      {task && <Box sx={{ borderTop: 1, borderBottom: 1, borderColor: 'divider', py: 0.75, px: 1 }}>
        <Stack direction="row" alignItems="center" spacing={0.75} flexWrap="wrap">
          <Chip size="small" label={task.status.replaceAll('_', ' ')} color={task.status === 'completed' ? 'success' : task.status === 'failed' ? 'error' : 'primary'} />
          <Typography variant="caption">Attempt {selectedRun?.attempt || 0} of {runs.length}</Typography>
          <IconButton size="small" disabled={runIndex <= 0} onClick={() => setRunIndex((value) => value - 1)}><NavigateBeforeIcon fontSize="small" /></IconButton>
          <IconButton size="small" disabled={runIndex < 0 || runIndex >= runs.length - 1} onClick={() => setRunIndex((value) => value + 1)}><NavigateNextIcon fontSize="small" /></IconButton>
          <Box sx={{ flex: 1 }} />
          {taskControls.map(({ action, label, availability }) => {
            return <Button
              key={action}
              size="small"
              color={action === 'cancel' ? 'error' : 'primary'}
              disabled={busy || !availability.enabled}
              title={availability.disabledReason}
              onClick={() => void command(action)}
            >{label}</Button>;
          })}
          <Button size="small" startIcon={<PsychologyIcon />} disabled={!selectedRun || !onOpenTrace} onClick={() => void openTrace()}>Debug Trace</Button>
        </Stack>
        <LinearProgress variant="determinate" value={task.progress} sx={{ mt: 0.75 }} />
      </Box>}
    </>}
    transcript={<ConversationTranscriptFrame>{items.map((item) => <TimelineBubble
      key={item.id}
      item={item}
      taskId={task?.id || ''}
      threadId={threadId}
      onSaveToMemory={onSaveToMemory}
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
    decision={invalidInterruptContract ? <Alert severity="error" sx={{ m: 2 }}>This human-input request has an invalid runtime response contract.</Alert> : pendingInterrupt && isApprovalInterrupt && responseOperation ? <Box sx={{ p: 2 }}>
      <Typography variant="subtitle2">{pendingInterrupt.title || 'Approval required'}</Typography>
      <Typography variant="body2" sx={{ my: 1 }}>{pendingInterrupt.description || pendingInterrupt.body}</Typography>
      <Stack direction="row" spacing={1} flexWrap="wrap">
        {(['once', 'session', 'always'] as const).map((choice) => <Button key={choice} size="small" variant="contained" disabled={Boolean(decisionSubmitting) || !isRuntimeOperationEnabled(effectiveSelectedRunCapabilities, responseOperation)} onClick={() => void decide('approve', { approvalScope: choice })}>Approve {choice}</Button>)}
        <Button size="small" color="error" disabled={Boolean(decisionSubmitting) || !isRuntimeOperationEnabled(effectiveSelectedRunCapabilities, responseOperation)} onClick={() => void decide('reject', { approvalScope: 'deny' })}>Deny</Button>
      </Stack>
      {decisionError ? <Alert severity="error" sx={{ mt: 1 }}>{decisionError}</Alert> : null}
    </Box> : pendingInterrupt && responseOperation ? <HumanReviewDecisionPanel
      interrupt={pendingInterrupt}
      submitting={decisionSubmitting}
      error={decisionError || null}
      disabled={!runtimeOperationAvailability(effectiveSelectedRunCapabilities, responseOperation).visible || !isRuntimeOperationEnabled(effectiveSelectedRunCapabilities, responseOperation)}
      disabledReason={runtimeOperationAvailability(effectiveSelectedRunCapabilities, responseOperation).disabledReason}
      scopeOptions={approvalScopeOptions}
      onAction={(action, options) => void decide(action, options)}
    /> : undefined}
    composer={!task ? <Box sx={{ pb: 1 }}>
      <ConversationComposer placeholder="Describe a new Deep Research objective…" busy={busy} disabled={!model || requestedWebUnavailable} onSubmit={(value) => void launch(value)} />
    </Box> : interactionDescriptors.length > 0 || stateUpdateAvailability.visible ? <Box sx={{ pb: 1 }}>
      <Stack direction="row" spacing={1} sx={{ mb: 1 }}>
        {interactionDescriptors.map((operation) => <Button
          key={operation.id}
          size="small"
          variant={interactionOperation === operation.id ? 'contained' : 'outlined'}
          disabled={!operation.availability.enabled}
          title={operation.availability.disabledReason}
          onClick={() => setInteractionOperation(operation.id)}
        >{operation.label}</Button>)}
        {stateUpdateAvailability.visible && <Button size="small" variant="outlined" disabled={!stateUpdateAvailability.enabled} title={stateUpdateAvailability.disabledReason} onClick={() => {
          const raw = window.prompt('State update JSON');
          if (!raw || !selectedRun) return;
          try {
            void updateAgentRunState(selectedRun.id, threadId, JSON.parse(raw)).then(() => refresh()).catch((reason) => setError(reason instanceof Error ? reason.message : String(reason)));
          } catch { setError('State update must be valid JSON.'); }
        }}>Update state</Button>}
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
    </Box> : <Box sx={{ px: 2, py: 1 }}><Typography variant="body2" color="text.secondary">
      {task.status === 'running' || task.status === 'queued' ? 'Research is running. You can pause or cancel it above.' : task.status === 'awaiting_approval' ? 'Review the approval request above to continue.' : task.status === 'paused' ? 'Research is paused. Resume or cancel it above.' : task.status === 'completed' ? 'This run is complete. Select New Deep Research task for a follow-up objective.' : 'Use the available lifecycle action above.'}
    </Typography></Box>}
  />;
}
