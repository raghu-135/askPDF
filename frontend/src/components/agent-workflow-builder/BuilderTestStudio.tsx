import React, { useEffect, useMemo, useRef, useState } from 'react';
import StopIcon from '@mui/icons-material/Stop';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import {
  Alert,
  Box,
  Button,
  Checkbox,
  Chip,
  CircularProgress,
  FormControlLabel,
  Paper,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import dynamic from 'next/dynamic';
import {
  cancelAgentWorkflowBuilderTest,
  getLatestAgentWorkflowBuilderTest,
  resumeAgentWorkflowBuilderTest,
  streamAgentWorkflowBuilderTest,
  type AgentRunDetails,
  type BuilderTestStreamEnvelope,
} from '../../lib/api';
import { AgentRunResumeAction, AgentRunStatus, InterruptStatus } from '../../lib/enums';
import { buildRunTraceView } from '../agent-debug/agent-trace-projection';
import {
  BuilderLlmModelPicker,
  BuilderThreadPicker,
  type BuilderModelHealth,
} from './BuilderTestPickers';

const AgentDebugCanvas = dynamic(() => import('../agent-graph/AgentDebugCanvas'), { ssr: false });

const getSessionId = () => {
  if (typeof window === 'undefined') return 'server';
  const key = 'askpdf-agent-workflow-builder-session';
  const existing = window.sessionStorage.getItem(key);
  if (existing) return existing;
  const next = globalThis.crypto?.randomUUID?.() || `builder-${Date.now()}`;
  window.sessionStorage.setItem(key, next);
  return next;
};

export default function BuilderTestStudio({
  spec,
  baseWorkflowId,
}: {
  spec: Record<string, any>;
  baseWorkflowId: string;
}) {
  const [threadId, setThreadId] = useState('');
  const [question, setQuestion] = useState('');
  const [llmModel, setLlmModel] = useState('');
  const [modelHealth, setModelHealth] = useState<BuilderModelHealth>({ checking: false, ready: null, supportsTools: null });
  const [useWeb, setUseWeb] = useState(false);
  const [externalConfirmed, setExternalConfirmed] = useState(false);
  const [running, setRunning] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [latest, setLatest] = useState<AgentRunDetails | null>(null);
  const [events, setEvents] = useState<BuilderTestStreamEnvelope[]>([]);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  const sessionId = useMemo(getSessionId, []);
  const traceView = latest ? buildRunTraceView(latest) : undefined;
  const liveTraceView = useMemo(() => {
    const nodeMap = new Map<string, any>();
    const tools: any[] = [];
    events.forEach((event) => {
      const nodeId = event.data?.node_id;
      if (nodeId && event.event.startsWith('node.')) {
        const status = event.event === 'node.started' ? 'active'
          : event.event === 'node.failed' ? 'error'
            : event.event === 'node.skipped' ? 'skipped'
              : 'completed';
        nodeMap.set(String(nodeId), {
          id: String(nodeId),
          type: event.data?.node_type,
          label: String(nodeId).replace(/_/g, ' '),
          instanceLabel: String(nodeId),
          visitIndex: event.data?.visit_index,
          status,
          skipped: status === 'skipped',
          raw: event.data,
        });
      }
      if (event.event === 'tool.completed') {
        tools.push({
          name: event.data?.tool_name || event.data?.name || 'tool',
          callerNode: event.data?.caller_node,
          ok: event.data?.ok !== false,
          warningCodes: event.data?.warnings || [],
          raw: event.data,
        });
      }
    });
    return {
      metrics: {},
      nodes: Array.from(nodeMap.values()),
      tools,
      usedNodeCount: nodeMap.size,
      usedToolCount: tools.length,
      warningCount: 0,
      errorCount: 0,
      errors: [],
    };
  }, [events]);
  const terminal = [...events].reverse().find((event) => event.event.startsWith('run.') && event.event !== 'run.started');
  const activeNode = [...events].reverse().find((event) => event.event === 'node.started')?.data?.node_id;
  const pending = latest?.pending_interrupt;
  const hasPendingInterrupt = Boolean(pending && String(pending.status || InterruptStatus.Pending) === InterruptStatus.Pending);
  const controlsLocked = running || hasPendingInterrupt;
  const lockedThreadId = hasPendingInterrupt ? latest?.thread_id || null : null;

  const refreshLatest = async () => {
    const result = await getLatestAgentWorkflowBuilderTest(sessionId, baseWorkflowId);
    setLatest(result);
    if (result) {
      setRunId(result.id);
      const resultPending = result.pending_interrupt;
      if (result.thread_id && resultPending && String(resultPending.status || InterruptStatus.Pending) === InterruptStatus.Pending) {
        setThreadId(result.thread_id);
      }
    }
  };

  useEffect(() => {
    void refreshLatest().catch(() => undefined);
  }, [baseWorkflowId, sessionId]);

  const acceptEvent = (event: BuilderTestStreamEnvelope) => {
    setEvents((current) => [...current.slice(-199), event]);
    if (event.data?.run_id) setRunId(String(event.data.run_id));
  };

  const runtime = {
    thread_id: threadId.trim(),
    llm_model: llmModel.trim(),
    use_web_search: useWeb,
    use_reranker: true,
    context_window: 4096,
    allow_external_tools: externalConfirmed,
    client_timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    client_locale: typeof navigator === 'undefined' ? undefined : navigator.language,
    client_now_iso: new Date().toISOString(),
  };

  const run = async () => {
    setError(null);
    setLatest(null);
    setEvents([]);
    setRunning(true);
    controller.current = new AbortController();
    try {
      await streamAgentWorkflowBuilderTest({
        ...runtime,
        builder_session_id: sessionId,
        base_workflow_id: baseWorkflowId,
        spec,
        question: question.trim(),
      }, acceptEvent, controller.current.signal);
      await refreshLatest();
    } catch (err: any) {
      if (err?.name !== 'AbortError') setError(err?.message || 'Test run failed.');
    } finally {
      setRunning(false);
      setStopping(false);
      controller.current = null;
    }
  };

  const stop = async () => {
    if (!runId) return;
    setStopping(true);
    try {
      await cancelAgentWorkflowBuilderTest(runId);
    } catch (err: any) {
      setError(err?.message || 'Unable to stop the test.');
      setStopping(false);
    }
  };

  const resume = async (action: 'approve' | 'reject') => {
    if (!runId || !pending?.interrupt_id) return;
    setRunning(true);
    setError(null);
    controller.current = new AbortController();
    try {
      await resumeAgentWorkflowBuilderTest(runId, {
        ...runtime,
        action,
        interrupt_id: pending.interrupt_id,
        resume_token: pending.resume_token || undefined,
        resume_version: pending.resume_version || undefined,
      }, acceptEvent, controller.current.signal);
      await refreshLatest();
    } catch (err: any) {
      if (err?.name !== 'AbortError') setError(err?.message || 'Unable to resume the test.');
    } finally {
      setRunning(false);
      controller.current = null;
    }
  };

  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', xl: '340px minmax(0, 1fr)' }, gap: 1.5 }}>
      <Paper variant="outlined" sx={{ p: 1.5, alignSelf: 'start' }}>
        <Typography variant="h6">Isolated test</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          Uses this unsaved graph and the thread&apos;s context. It does not add chat messages, memory, statistics, or change thread settings.
        </Typography>
        <Stack spacing={1.25}>
          <BuilderThreadPicker value={threadId} onChange={setThreadId} disabled={controlsLocked} lockedThreadId={lockedThreadId} />
          <BuilderLlmModelPicker
            value={llmModel}
            onChange={(model) => {
              setLlmModel(model);
              setModelHealth({ checking: Boolean(model), ready: null, supportsTools: null });
            }}
            onHealthChange={setModelHealth}
            disabled={controlsLocked}
          />
          <TextField label="Test question" multiline minRows={3} value={question} onChange={(event) => setQuestion(event.target.value)} required disabled={controlsLocked} />
          <FormControlLabel control={<Checkbox checked={useWeb} disabled={controlsLocked} onChange={(event) => setUseWeb(event.target.checked)} />} label="Allow web search" />
          {useWeb && (
            <FormControlLabel
              control={<Checkbox checked={externalConfirmed} disabled={controlsLocked} onChange={(event) => setExternalConfirmed(event.target.checked)} />}
              label="I confirm this test may call external tools"
            />
          )}
          <Stack direction="row" spacing={1}>
            <Button
              variant="contained"
              startIcon={running ? <CircularProgress size={15} color="inherit" /> : <PlayArrowIcon />}
              disabled={controlsLocked || !threadId.trim() || !llmModel.trim() || modelHealth.checking || modelHealth.ready !== true || !question.trim() || (useWeb && !externalConfirmed)}
              onClick={() => void run()}
            >
              Run unsaved graph
            </Button>
            {running && <Button color="error" startIcon={<StopIcon />} disabled={!runId || stopping} onClick={() => void stop()}>{stopping ? 'Stopping…' : 'Stop'}</Button>}
          </Stack>
        </Stack>
        {error && <Alert severity="error" sx={{ mt: 1.5 }}>{error}</Alert>}
      </Paper>
      <Stack spacing={1.5} sx={{ minWidth: 0 }}>
        <Paper variant="outlined" sx={{ p: 1.5 }}>
          <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
            <Typography variant="h6">Live progress</Typography>
            <Chip size="small" label={latest?.status || terminal?.event || (running ? 'running' : 'not run')} color={running ? 'primary' : 'default'} />
            {activeNode && running && <Chip size="small" variant="outlined" label={`Active: ${activeNode}`} />}
          </Stack>
          {events.length === 0 && !latest ? (
            <Typography variant="body2" color="text.secondary">Start a test to see nodes, routes, and tools as they execute.</Typography>
          ) : (
            <Stack direction="row" spacing={0.75} sx={{ flexWrap: 'wrap', rowGap: 0.75 }}>
              {events.filter((event) => event.event !== 'heartbeat').slice(-14).map((event, index) => (
                <Chip key={`${event.id}-${index}`} size="small" variant="outlined" label={`${event.event}${event.data?.node_id ? ` · ${event.data.node_id}` : ''}`} />
              ))}
            </Stack>
          )}
          {terminal?.data?.answer && <Typography sx={{ mt: 1.5, whiteSpace: 'pre-wrap' }}>{terminal.data.answer}</Typography>}
        </Paper>
        {pending && String(pending.status || InterruptStatus.Pending) === InterruptStatus.Pending && (
          <Alert severity="info" action={
            <Stack direction="row" spacing={0.5}>
              <Button size="small" onClick={() => void resume(AgentRunResumeAction.Approve)}>Approve</Button>
              <Button size="small" color="error" onClick={() => void resume(AgentRunResumeAction.Reject)}>Reject</Button>
            </Stack>
          }>
            {pending.title || pending.prompt || 'This step needs human approval.'}
          </Alert>
        )}
        {(latest || events.length > 0) && (
          <Box sx={{ minHeight: 520 }}>
            <AgentDebugCanvas
              resolvedSpec={latest?.resolved_spec_json || spec}
              workflowId={latest?.workflow_id || baseWorkflowId}
              traceView={traceView || liveTraceView}
            />
          </Box>
        )}
        {latest?.status === AgentRunStatus.Cancelled && <Alert severity="warning">The test stopped after the active node finished. Its partial trace is shown above.</Alert>}
      </Stack>
    </Box>
  );
}
