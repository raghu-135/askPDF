import React, { useEffect, useMemo, useRef, useState } from 'react';
import StopIcon from '@mui/icons-material/Stop';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import {
  Alert,
  Box,
  Button,
  Checkbox,
  CircularProgress,
  FormControlLabel,
  Paper,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import {
  cancelAgentWorkflowBuilderTest,
  getLatestAgentWorkflowBuilderTest,
  resumeAgentWorkflowBuilderTest,
  streamAgentWorkflowBuilderTest,
  type AgentRunDetails,
  type BuilderTestStreamEnvelope,
} from '../../lib/api';
import { AgentRunResumeAction, AgentRunStatus, InterruptStatus } from '../../lib/enums';
import { buildLiveTraceView, buildRunTraceView } from '../agent-debug/agent-trace-projection';
import AgentExecutionView from '../agent-graph/AgentExecutionView';
import {
  BuilderLlmModelPicker,
  BuilderThreadPicker,
  type BuilderModelHealth,
} from './BuilderTestPickers';

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
  const retainedTraceView = useMemo(() => latest ? buildRunTraceView(latest) : undefined, [latest]);
  const liveTraceView = useMemo(() => buildLiveTraceView(events), [events]);
  const executionTraceView = useMemo(() => {
    if (!running || events.length === 0) return retainedTraceView || liveTraceView;
    if (!retainedTraceView) return liveTraceView;
    const liveVisits = new Set(liveTraceView.nodes.map((node) => `${node.id}:${node.visitIndex || 1}`));
    const missingRetainedNodes = retainedTraceView.nodes.filter((node) => !liveVisits.has(`${node.id}:${node.visitIndex || 1}`));
    if (missingRetainedNodes.length === 0) return liveTraceView;
    const nodes = [...missingRetainedNodes, ...liveTraceView.nodes];
    const tools = [...retainedTraceView.tools, ...liveTraceView.tools];
    const detailManifest = new Map(
      [...retainedTraceView.detailManifest, ...liveTraceView.detailManifest]
        .map((row) => [`${row.node_id}:${row.visit_index}`, row] as const),
    );
    return {
      ...retainedTraceView,
      ...liveTraceView,
      nodes,
      tools,
      usedNodeCount: new Set(nodes.filter((node) => !node.skipped).map((node) => node.id)).size,
      usedToolCount: tools.length,
      warningCount: nodes.reduce((count, node) => count + node.warningCodes.length, 0)
        + tools.reduce((count, tool) => count + tool.warningCodes.length, 0),
      errorCount: nodes.filter((node) => node.status === 'error').length
        + tools.filter((tool) => !tool.ok).length,
      errors: nodes.map((node) => node.error).filter((nodeError) => Boolean(nodeError && Object.keys(nodeError).length)),
      finalOutput: liveTraceView.finalOutput || retainedTraceView.finalOutput,
      detailManifest: [...detailManifest.values()],
    };
  }, [events.length, liveTraceView, retainedTraceView, running]);
  const terminal = [...events].reverse().find((event) => event.event.startsWith('run.') && event.event !== 'run.started');
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
    if (event.event !== 'heartbeat') setEvents((current) => [...current, event]);
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
    setRunId(null);
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
      setStopping(false);
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
        {pending && String(pending.status || InterruptStatus.Pending) === InterruptStatus.Pending && (
          <Alert severity="info" action={
            <Stack direction="row" spacing={0.5}>
              <Button size="small" disabled={running} onClick={() => void resume(AgentRunResumeAction.Approve)}>Approve</Button>
              <Button size="small" color="error" disabled={running} onClick={() => void resume(AgentRunResumeAction.Reject)}>Reject</Button>
            </Stack>
          }>
            {pending.title || pending.prompt || 'This step needs human approval.'}
          </Alert>
        )}
        {(latest || events.length > 0) && (
          <AgentExecutionView
            runId={runId}
            threadId={latest?.thread_id || threadId}
            resolvedSpec={latest?.resolved_spec_json || spec}
            workflowId={latest?.workflow_id || baseWorkflowId}
            traceView={executionTraceView}
            status={latest?.status || terminal?.event || (running ? 'running' : 'not run')}
            running={running}
          />
        )}
        {latest?.status === AgentRunStatus.Cancelled && <Alert severity="warning">The test stopped after the active node finished. Its partial trace is shown above.</Alert>}
      </Stack>
    </Box>
  );
}
