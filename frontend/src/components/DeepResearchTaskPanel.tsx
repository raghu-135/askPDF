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
  commandAgentTask,
  createAgentTask,
  deleteAgentTask,
  downloadAgentTaskArtifact,
  getDeepResearchCapabilities,
  getAgentRun,
  getAgentTask,
  getAgentTaskRuns,
  getAgentTaskTodos,
  getAgentTaskTimeline,
  listAgentTasks,
  resumeAgentRun,
  steerAgentTask,
  type AgentTaskRun,
  type AgentTaskSummary,
  type AgentTaskTimelineItem,
  type AgentTaskTodo,
  type AgentRunResumeAction,
  type DeepResearchEngine,
} from '../lib/api';
import { mergeActiveAgentTaskRun, resolveDeepResearchContextWindow, shouldPollAgentTask } from '../lib/deep-research-ui-state';
import {
  deriveConversationSentences,
  type ConversationSentence,
  type ConversationSentenceCache,
} from '../lib/chat-sentence-cache';
import type { ChatTraceDescriptor } from './ChatInterface';
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
  const [webCapability, setWebCapability] = useState<boolean | null>(null);
  const [engine, setEngine] = useState<DeepResearchEngine>('langgraph');
  const [hermesEnabled, setHermesEnabled] = useState(false);
  const [hermesMaxContext, setHermesMaxContext] = useState<number | null>(null);
  const [capabilityError, setCapabilityError] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const lastSequence = useRef(0);
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
    setCapabilityError('');
    void getDeepResearchCapabilities()
      .then((capabilities) => {
        if (!active) return;
        setWebCapability(capabilities.web_enabled);
        setHermesEnabled(Boolean(capabilities.engines?.hermes?.enabled));
        setHermesMaxContext(capabilities.engines?.hermes?.max_context_length ?? null);
      })
      .catch(() => {
        if (!active) return;
        setWebCapability(false);
        setCapabilityError('Deep Research capabilities could not be loaded. Internet research is unavailable until the service recovers.');
      });
    return () => { active = false; };
  }, []);

  const refresh = useCallback(async () => {
    if (!selectedTaskId) { setTask(null); setRuns([]); setTodos([]); setItems([]); setRunIndex(-1); return; }
    const [nextTask, fetchedRuns, nextTodos] = await Promise.all([
      getAgentTask(selectedTaskId, threadId),
      getAgentTaskRuns(selectedTaskId, threadId),
      getAgentTaskTodos(selectedTaskId, threadId),
    ]);
    const nextRuns = mergeActiveAgentTaskRun(nextTask, fetchedRuns);
    setTask(nextTask);
    setRuns(nextRuns);
    setTodos(nextTodos);
    setRunIndex((current) => current >= 0 && current < nextRuns.length ? current : nextRuns.length - 1);
  }, [selectedTaskId, threadId]);

  useEffect(() => { setError(''); void refresh().catch((value) => setError(String(value))); }, [refresh]);
  useEffect(() => {
    if (!shouldPollAgentTask(task)) return;
    let cancelled = false;
    let timer: number | undefined;
    const poll = async () => {
      try { await refresh(); }
      catch (value) { if (!cancelled) setError(String(value)); }
      if (!cancelled) timer = window.setTimeout(poll, 2000);
    };
    timer = window.setTimeout(poll, 2000);
    return () => { cancelled = true; if (timer !== undefined) window.clearTimeout(timer); };
  }, [refresh, task?.status]);
  useEffect(() => {
    if (!selectedTaskId || !selectedRun) { setItems([]); return; }
    void getAgentTaskTimeline(selectedTaskId, selectedRun.id, threadId).then((value) => { setTask(value.task); setItems(value.items); }).catch((value) => setError(String(value)));
  }, [selectedRun?.id, selectedTaskId, threadId]);

  useEffect(() => {
    if (!selectedTaskId || !selectedRun) return;
    lastSequence.current = 0;
    const query = new URLSearchParams({ thread_id: threadId, run_id: selectedRun.id, scope: 'run', after_sequence: String(lastSequence.current) });
    const source = new EventSource(`${API_BASE}/api/agent-tasks/${encodeURIComponent(selectedTaskId)}/events?${query}`);
    source.addEventListener('task_event', (event) => {
      lastSequence.current = Math.max(lastSequence.current, Number(event.lastEventId || 0));
      let payload: any = {};
      try { payload = JSON.parse((event as MessageEvent).data || '{}'); } catch { payload = {}; }
      const type = String(payload.type || '');
      if (type.startsWith('task.') || type.startsWith('todo.') || type.startsWith('subagent.')) void refresh();
      if (/^(plan\.|todo\.|subagent\.|artifact\.|task\.approval|task\.(completed|failed|cancelled))/.test(type)) {
        void getAgentTaskTimeline(selectedTaskId, selectedRun.id, threadId).then((value) => setItems(value.items));
      }
    });
    return () => source.close();
  }, [selectedRun?.id, selectedTaskId, threadId, refresh]);

  const launch = async (objective: string) => {
    if (webSearchMode !== 'off' && webCapability !== true) {
      setError('Internet research is not available for Deep Research. Switch Internet Search off and try again.');
      return;
    }
    setBusy(true); setError('');
    try {
      const effectiveContextWindow = resolveDeepResearchContextWindow(engine, contextWindow, hermesMaxContext);
      const created = await createAgentTask(threadId, { objective, llm_model: model, context_window: effectiveContextWindow, web_search_mode: webSearchMode, engine });
      const started = await commandAgentTask(created.id, threadId, 'start', created.version);
      onTaskSelect(started.id);
    } catch (value) { setError(value instanceof Error ? value.message : String(value)); }
    finally { setBusy(false); }
  };

  const command = async (action: 'pause' | 'resume' | 'cancel' | 'retry') => {
    if (!task) return;
    setBusy(true); setError('');
    try { setTask(await commandAgentTask(task.id, threadId, action, task.version)); await refresh(); }
    catch (value) { setError(value instanceof Error ? value.message : String(value)); await refresh(); }
    finally { setBusy(false); }
  };

  const openTrace = async () => {
    if (!selectedRun || !onOpenTrace) return;
    const details = await getAgentRun(selectedRun.id, threadId);
    onOpenTrace({ id: selectedRun.id, messageId: `agent-task:${task?.id}:${selectedRun.id}`, label: `Deep Research · attempt ${selectedRun.attempt}`, status: selectedRun.status, runDetails: details });
  };

  const decide = async (
    action: AgentRunResumeAction,
    options?: { selectedOptionIds?: string[]; editedPayload?: Record<string, unknown>; runtimeApprovalChoice?: 'once' | 'session' | 'always' | 'deny' },
  ) => {
    const pending = selectedRun?.pending_interrupt;
    if (!selectedRun || !pending) return;
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
        runtime_approval_choice: options?.runtimeApprovalChoice,
      });
      await refresh();
    } catch (value) { setDecisionError(value instanceof Error ? value.message : String(value)); }
    finally { setDecisionSubmitting(null); }
  };

  const actions = useMemo(() => {
    if (!task) return [] as Array<'pause' | 'resume' | 'cancel' | 'retry'>;
    if (task.status === 'running' || task.status === 'queued') return ['pause', 'cancel'] as const;
    if (task.status === 'paused') return ['resume', 'cancel'] as const;
    if (task.status === 'awaiting_approval') return ['cancel'] as const;
    if (task.status === 'failed' || task.status === 'expired') return ['retry'] as const;
    return [] as Array<'pause' | 'resume' | 'cancel' | 'retry'>;
  }, [task]);
  const frozen = Boolean(selectedRun);
  const frozenEngine: DeepResearchEngine = task?.workflow_id === 'hermes_rag_agent' ? 'hermes' : 'langgraph';
  const displayedContextWindow = frozen
    ? Number(task?.configuration?.context_window || contextWindow)
    : resolveDeepResearchContextWindow(engine, contextWindow, hermesMaxContext);
  const configuredWebMode = String(task?.configuration?.web_search_mode || 'off') as 'off' | 'ask' | 'on';
  const frozenWebMode = task?.web_access === 'allowed_for_task'
    ? 'on'
    : task?.web_access === 'denied_for_task'
      ? 'off'
      : configuredWebMode;
  const requestedWebUnavailable = !frozen && webSearchMode !== 'off' && webCapability !== true;
  const pendingInterrupt = selectedRun?.pending_interrupt?.status === 'pending' ? selectedRun.pending_interrupt : null;
  const isHermesApproval = frozenEngine === 'hermes' && pendingInterrupt?.type === 'hermes_approval';
  const approvalTodoIds = Array.isArray(pendingInterrupt?.approval_scope?.todo_ids)
    ? pendingInterrupt.approval_scope.todo_ids.map(String)
    : [];
  const approvalScopeOptions = approvalTodoIds.map((id) => {
    const todo = todos.find((item) => item.id === id);
    return { id, label: todo?.title || id, description: todo?.description };
  });
  useEffect(() => setDecisionError(''), [pendingInterrupt?.interrupt_id]);

  return <ConversationPanelTemplate
    sx={{ p: 1 }}
    header={<ConversationHeader
      models={models}
      model={frozen ? String(task?.configuration?.llm_model || model) : model}
      contextWindow={displayedContextWindow}
      disabled={frozen}
      contextWindowDisabled={!frozen && engine === 'hermes'}
      onModelChange={onModelChange}
      onContextWindowChange={onContextWindowChange}
      leading={<><Tooltip title="Back to chat"><IconButton size="small" onClick={onBack}><ArrowBackIcon fontSize="small" /></IconButton></Tooltip>{embeddingControl}</>}
      beforeModelControls={<Stack direction="row" spacing={1} alignItems="center">
        <Chip
          size="small"
          label={(frozen ? frozenEngine : engine) === 'hermes' ? 'Hermes' : 'LangGraph'}
          color={(frozen ? frozenEngine : engine) === 'hermes' ? 'secondary' : 'default'}
          onClick={frozen ? undefined : () => setEngine((value) => value === 'langgraph' && hermesEnabled ? 'hermes' : 'langgraph')}
          title={!hermesEnabled && !frozen ? 'Hermes is not configured' : 'Deep Research engine'}
        />
        {engine === 'hermes' && !frozen && hermesMaxContext !== null
          ? <Typography variant="caption" color="text.secondary">Hermes env: {hermesMaxContext.toLocaleString()}</Typography>
          : null}
        {renderWebControl(frozen ? frozenWebMode : webSearchMode, frozen || webCapability === false)}
      </Stack>}
      trailingActions={<DeepResearchTaskPicker threadId={threadId} selectedTaskId={selectedTaskId} onSelect={onTaskSelect} />}
    />}
    status={<>
      {error && <Alert severity="error" sx={{ mb: 1 }}>{error}</Alert>}
      {capabilityError && <Alert severity="warning" sx={{ mb: 1 }}>{capabilityError}</Alert>}
      {task?.terminal_reason === 'required_evidence_unavailable' && <Alert severity="error" sx={{ mb: 1 }}>Hermes could not retrieve the evidence required for this report. The generated text was not published as a grounded result.</Alert>}
      {task && <Box sx={{ borderTop: 1, borderBottom: 1, borderColor: 'divider', py: 0.75, px: 1 }}>
        <Stack direction="row" alignItems="center" spacing={0.75} flexWrap="wrap">
          <Chip size="small" label={task.status.replaceAll('_', ' ')} color={task.status === 'completed' ? 'success' : task.status === 'failed' ? 'error' : 'primary'} />
          <Typography variant="caption">Attempt {selectedRun?.attempt || 0} of {runs.length}</Typography>
          <IconButton size="small" disabled={runIndex <= 0} onClick={() => setRunIndex((value) => value - 1)}><NavigateBeforeIcon fontSize="small" /></IconButton>
          <IconButton size="small" disabled={runIndex < 0 || runIndex >= runs.length - 1} onClick={() => setRunIndex((value) => value + 1)}><NavigateNextIcon fontSize="small" /></IconButton>
          <Box sx={{ flex: 1 }} />
          {actions.map((action) => <Button key={action} size="small" color={action === 'cancel' ? 'error' : 'primary'} disabled={busy} onClick={() => void command(action)}>{action}</Button>)}
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
    decision={pendingInterrupt && isHermesApproval ? <Box sx={{ p: 2 }}>
      <Typography variant="subtitle2">{pendingInterrupt.title || 'Hermes approval required'}</Typography>
      <Typography variant="body2" sx={{ my: 1 }}>{pendingInterrupt.description || pendingInterrupt.body}</Typography>
      <Stack direction="row" spacing={1} flexWrap="wrap">
        {(['once', 'session', 'always'] as const).map((choice) => <Button key={choice} size="small" variant="contained" disabled={Boolean(decisionSubmitting)} onClick={() => void decide('approve', { runtimeApprovalChoice: choice })}>Approve {choice}</Button>)}
        <Button size="small" color="error" disabled={Boolean(decisionSubmitting)} onClick={() => void decide('reject', { runtimeApprovalChoice: 'deny' })}>Deny</Button>
      </Stack>
      {decisionError ? <Alert severity="error" sx={{ mt: 1 }}>{decisionError}</Alert> : null}
    </Box> : pendingInterrupt ? <HumanReviewDecisionPanel
      interrupt={pendingInterrupt}
      submitting={decisionSubmitting}
      error={decisionError || null}
      scopeOptions={approvalScopeOptions}
      onAction={(action, options) => void decide(action, options)}
    /> : undefined}
    composer={!task ? <Box sx={{ pb: 1 }}>
      <ConversationComposer placeholder="Describe a new Deep Research objective…" busy={busy} disabled={!model || requestedWebUnavailable} onSubmit={(value) => void launch(value)} />
    </Box> : frozenEngine === 'hermes' && task.status === 'running' ? <Box sx={{ pb: 1 }}>
      <ConversationComposer placeholder="Steer the active Hermes run…" busy={busy} onSubmit={async (value) => {
        setBusy(true); setError('');
        try { await steerAgentTask(task.id, threadId, value, task.version); await refresh(); }
        catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
        finally { setBusy(false); }
      }} />
    </Box> : <Box sx={{ px: 2, py: 1 }}><Typography variant="body2" color="text.secondary">
      {task.status === 'running' || task.status === 'queued' ? 'Research is running. You can pause or cancel it above.' : task.status === 'awaiting_approval' ? 'Review the approval request above to continue.' : task.status === 'paused' ? 'Research is paused. Resume or cancel it above.' : task.status === 'completed' ? 'This run is complete. Select New Deep Research task for a follow-up objective.' : 'Use the available lifecycle action above.'}
    </Typography></Box>}
  />;
}
