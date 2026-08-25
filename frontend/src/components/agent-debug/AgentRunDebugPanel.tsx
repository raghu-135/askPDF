import React, { useEffect, useMemo, useRef, useState } from 'react';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DownloadIcon from '@mui/icons-material/Download';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { Box, Button, Checkbox, Chip, CircularProgress, FormControlLabel, IconButton, Tooltip, Typography } from '@mui/material';
import { getAgentRun, getAgentRunCapabilities, resumeAgentRun, type AgentRunDetails, type AgentRunResumeAction, type AgentRuntimeCapabilityResponse, type AgentTraceRefs } from '../../lib/api';
import { isRuntimeOperationEnabled, runtimeInterruptResponseOperation, runtimeOperationAvailability } from '../../lib/runtime-capabilities';
import { AgentRunResumeAction as AgentRunResumeActionValue, AgentRunStatus, HitlSelectionMode, InterruptStatus } from '../../lib/enums';
import { buildCorrectiveInspection, buildRunTraceView, buildTraceExportJson, getRetainedRunErrorMessage, mergeLiveAndRetainedTraceViews, shouldRefreshRetainedTrace, type TraceRunView } from './agent-trace-projection';
import AgentExecutionView from '../agent-graph/AgentExecutionView';
import { compactExecutionText } from '../agent-graph/agent-execution-display';
import { PARALLEL_WORKER_STATUS_LABELS } from '../../lib/parallel-runtime';
import { isTaskOwnedAgentRun } from '../../lib/deep-research-ui-state';
import { withRetry } from '../../lib/retry-utils';

function AgentRunDebugPanel({
  runId,
  threadId,
  routeReason,
  traceRefs,
  runDetails: providedRunDetails,
  loading,
  error,
  onRunDetailsChange,
  suspendHeavyContent = false,
  liveTraceView,
  running = false,
  onResumeAction,
}: {
  runId: string;
  threadId?: string;
  routeReason?: string;
  traceRefs?: AgentTraceRefs | null;
  runDetails?: AgentRunDetails;
  loading?: boolean;
  error?: string;
  onRunDetailsChange?: (runDetails: AgentRunDetails) => void;
  suspendHeavyContent?: boolean;
  liveTraceView?: TraceRunView;
  running?: boolean;
  onResumeAction?: (action: AgentRunResumeAction, selectedOptionIds?: string[]) => Promise<boolean>;
}) {
  const [copyStatus, setCopyStatus] = useState<'idle' | 'copied' | 'failed'>('idle');
  const [resumeSubmitting, setResumeSubmitting] = useState<AgentRunResumeAction | null>(null);
  const [resumeError, setResumeError] = useState<string | null>(null);
  const [resumeMessage, setResumeMessage] = useState<string | null>(null);
  const [selectedOptionIds, setSelectedOptionIds] = useState<string[]>([]);
  const resumeSubmissionKeyRef = useRef<string | null>(null);
  const traceRefreshAttemptedRef = useRef(new Map<string, number>());
  const [traceRefreshExhausted, setTraceRefreshExhausted] = useState(false);
  const [refreshedRunDetails, setRefreshedRunDetails] = useState<AgentRunDetails | undefined>();
  const [runCapabilities, setRunCapabilities] = useState<AgentRuntimeCapabilityResponse | null>(null);
  const runDetails = refreshedRunDetails?.id === runId ? refreshedRunDetails : providedRunDetails;
  const debug = runDetails?.debug;
  const retainedErrorMessage = runDetails ? getRetainedRunErrorMessage(runDetails) : null;
  const pendingInterrupt = runDetails?.pending_interrupt;
  const isTaskOwnedRun = isTaskOwnedAgentRun(runDetails);
  const interruptStatus = pendingInterrupt?.status || (pendingInterrupt ? InterruptStatus.Pending : undefined);
  const allowedActions = Array.isArray(pendingInterrupt?.allowed_actions)
    ? pendingInterrupt.allowed_actions.map(String)
    : [];
  const responseOperation = runtimeInterruptResponseOperation(pendingInterrupt);
  const responseAvailability = responseOperation
    ? runtimeOperationAvailability(runCapabilities, responseOperation)
    : { visible: false, enabled: false, disabledReason: 'invalid_interrupt_response_operation' };
  const traceView = useMemo(() => runDetails ? buildRunTraceView(runDetails) : undefined, [runDetails]);
  const unsupportedTraceFormat = Boolean(
    runDetails?.debug
    && (runDetails.debug.version !== 2 || !runDetails.debug.diagnostics),
  );
  const executionTraceView = useMemo(
    () => liveTraceView ? mergeLiveAndRetainedTraceViews(liveTraceView, traceView) : traceView,
    [liveTraceView, traceView],
  );
  const trace = traceView?.trace;
  const traceJson = useMemo(() => buildTraceExportJson(traceView), [traceView]);
  const interruptOptions = Array.isArray(pendingInterrupt?.options)
    ? pendingInterrupt.options.filter((option) => option && typeof option.id === 'string')
    : [];
  const selectionMode = String(pendingInterrupt?.selection_mode || HitlSelectionMode.Single);
  const isMultiSelect = selectionMode === HitlSelectionMode.Multi || selectionMode === HitlSelectionMode.SingleOrMulti;
  const executionThreadId = runDetails?.thread_id || null;
  const executionWorkflowId = runDetails?.workflow_id;
  const executionFramework = (runDetails as any)?.framework || (runDetails as any)?.runtime_metadata?.framework;
  const executionResolvedSpec = runDetails?.resolved_spec_json;
  const executionStatus = runDetails?.status || (running ? 'running' : undefined);
  const executionDetailsAvailable = Boolean(runDetails && !running);
  const parallelSummary = liveTraceView?.parallel?.summary || runDetails?.parallel_summary || runDetails?.metrics_json?.parallel_summary;
  const parallelTasks = liveTraceView?.parallel?.tasks || [];
  const correctiveInspection = useMemo(
    () => runDetails ? buildCorrectiveInspection(runDetails, traceView?.metrics) : undefined,
    [runDetails, traceView?.metrics],
  );
  const corrective = correctiveInspection?.corrective;
  const retrievalQuality = correctiveInspection?.retrievalQuality;
  const grounding = correctiveInspection?.grounding;
  const groundingUsefulness = grounding?.usefulness
    || (typeof grounding?.usefulness_score === 'number'
      ? (Number(grounding.usefulness_score) >= 3 ? 'yes (historical score)' : 'no (historical score)')
      : undefined);

  useEffect(() => {
    setRefreshedRunDetails(undefined);
    setTraceRefreshExhausted(false);
  }, [runId, providedRunDetails]);

  useEffect(() => {
    const metadataThreadId = threadId || runDetails?.thread_id;
    if (runDetails || !metadataThreadId) return undefined;
    let active = true;
    void getAgentRun(runId, metadataThreadId)
      .then((details) => {
        if (!active) return;
        setRefreshedRunDetails(details);
        onRunDetailsChange?.(details);
      })
      .catch(() => {
        // The live event projection remains usable when metadata is not yet
        // available; the existing loading/retained-state handling reports the
        // final result when polling catches up.
      });
    return () => {
      active = false;
    };
  }, [onRunDetailsChange, runDetails, runId, threadId]);

  useEffect(() => {
    let active = true;
    const threadId = runDetails?.thread_id;
    setRunCapabilities(null);
    if (!threadId) return () => { active = false; };
    void withRetry(() => getAgentRunCapabilities(runId, threadId), { maxRetries: 3, baseDelay: 500 })
      .then((result) => {
        if (!active) return;
        setRunCapabilities(result.success ? result.data || null : null);
      });
    return () => { active = false; };
  }, [pendingInterrupt?.interrupt_id, pendingInterrupt?.status, pendingInterrupt?.resume_version, pendingInterrupt?.response_operation, runDetails?.status, runDetails?.runtime_binding_status, runDetails?.thread_id, runId]);

  useEffect(() => {
    if (!runDetails || !executionThreadId || !shouldRefreshRetainedTrace(runDetails)) return;
    const attempts = traceRefreshAttemptedRef.current.get(runId) || 0;
    if (attempts >= 5) {
      setTraceRefreshExhausted(true);
      return;
    }
    traceRefreshAttemptedRef.current.set(runId, attempts + 1);
    const timer = window.setTimeout(() => {
      void getAgentRun(runId, executionThreadId)
        .then((refreshed) => {
          setRefreshedRunDetails(refreshed);
          onRunDetailsChange?.(refreshed);
        })
        .catch(() => undefined);
    }, 500 * (attempts + 1));
    return () => window.clearTimeout(timer);
  }, [executionThreadId, onRunDetailsChange, runDetails, runId]);

  useEffect(() => {
    if (interruptOptions.length === 0) {
      setSelectedOptionIds([]);
      return;
    }
    setSelectedOptionIds((current) => {
      const valid = current.filter((id) => interruptOptions.some((option) => option.id === id));
      if (valid.length > 0) return isMultiSelect ? valid : valid.slice(0, 1);
      return [interruptOptions[0].id];
    });
  }, [pendingInterrupt?.interrupt_id, isMultiSelect, interruptOptions.map((option) => option.id).join('|')]);

  const copyTrace = async () => {
    if (!traceJson || typeof navigator === 'undefined' || !navigator.clipboard) return;
    try {
      await navigator.clipboard.writeText(traceJson);
      setCopyStatus('copied');
      window.setTimeout(() => setCopyStatus('idle'), 1600);
    } catch {
      setCopyStatus('failed');
      window.setTimeout(() => setCopyStatus('idle'), 1600);
    }
  };

  const downloadTrace = () => {
    if (!traceJson || typeof window === 'undefined') return;
    const blob = new Blob([traceJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `agent-trace-${runId}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleResume = async (action: AgentRunResumeAction) => {
    if (!runDetails || !pendingInterrupt?.interrupt_id) return;
    if (!responseOperation) {
      setResumeError('This human-input request has an invalid runtime response contract.');
      return;
    }
    const submissionKey = `${runId}:${pendingInterrupt.interrupt_id}:${pendingInterrupt.resume_version ?? 1}:${action}`;
    if (resumeSubmissionKeyRef.current) return;
    resumeSubmissionKeyRef.current = submissionKey;
    setResumeSubmitting(action);
    setResumeError(null);
    setResumeMessage(null);
    try {
      if (onResumeAction) {
        const resumed = await onResumeAction(action, action === AgentRunResumeActionValue.ApproveSelected ? selectedOptionIds : undefined);
        if (!resumed) throw new Error('Unable to submit decision.');
        setResumeMessage('Decision applied.');
        return;
      }
      if (!executionThreadId) {
        throw new Error('Cannot submit human review because the run thread is unavailable.');
      }
      const response = await resumeAgentRun(runId, {
        action,
        interrupt_id: pendingInterrupt.interrupt_id,
        resume_token: pendingInterrupt.resume_token || undefined,
        resume_version: pendingInterrupt.resume_version || undefined,
        thread_id: executionThreadId,
        selected_option_ids: action === AgentRunResumeActionValue.ApproveSelected ? selectedOptionIds : undefined,
        client_metadata: { source: 'agent_run_debug_panel' },
      });
      onRunDetailsChange?.(response.agent_run);
      const status = response.agent_run?.status;
      setResumeMessage(
        response.duplicate
          ? 'Decision already recorded.'
          : status === AgentRunStatus.Completed || status === AgentRunStatus.Clarification
            ? 'Decision applied. Run resumed.'
            : status === AgentRunStatus.Failed
              ? 'Decision applied. Resume failed.'
              : 'Decision applied.'
      );
    } catch (err: any) {
      setResumeError(err?.message || 'Unable to submit decision.');
    } finally {
      if (resumeSubmissionKeyRef.current === submissionKey) {
        resumeSubmissionKeyRef.current = null;
      }
      setResumeSubmitting(null);
    }
  };

  const renderInterruptAction = (
    action: AgentRunResumeAction,
    label: string,
    icon: React.ReactNode,
    color: 'primary' | 'error' | 'inherit' = 'primary',
  ) => {
    if (!allowedActions.includes(action) || !isRuntimeOperationEnabled(runCapabilities, responseOperation)) return null;
    return (
      <Button
        key={action}
        size="small"
        variant={action === AgentRunResumeActionValue.Reject ? 'outlined' : 'contained'}
        color={color === 'inherit' ? undefined : color}
        startIcon={icon}
        disabled={Boolean(resumeSubmitting)}
        onClick={() => handleResume(action)}
      >
        {resumeSubmitting === action ? 'Submitting...' : label}
      </Button>
    );
  };

  const toggleOption = (optionId: string) => {
    setSelectedOptionIds((current) => {
      if (!isMultiSelect) return [optionId];
      if (current.includes(optionId)) {
        const next = current.filter((id) => id !== optionId);
        return next.length > 0 ? next : current;
      }
      return [...current, optionId];
    });
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', width: '100%', minWidth: 0, maxWidth: '100%', overflowX: 'hidden' }}>
      <Box sx={{ px: 1, py: 0.5, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 0.5, bgcolor: 'background.paper', borderBottom: 1, borderColor: 'divider' }}>
        <Tooltip title={runId} arrow>
          <Typography variant="caption" color="text.secondary">
            Run …{runId.slice(-8)}
          </Typography>
        </Tooltip>
        {trace && (
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Tooltip title={copyStatus === 'copied' ? 'Copied trace JSON' : copyStatus === 'failed' ? 'Copy failed' : 'Copy trace JSON'} arrow>
              <span>
                <IconButton size="small" onClick={copyTrace} disabled={!traceJson} aria-label="Copy trace JSON">
                  <ContentCopyIcon fontSize="inherit" />
                </IconButton>
              </span>
            </Tooltip>
            <Tooltip title="Download trace JSON" arrow>
              <span>
                <IconButton size="small" onClick={downloadTrace} disabled={!traceJson} aria-label="Download trace JSON">
                  <DownloadIcon fontSize="inherit" />
                </IconButton>
              </span>
            </Tooltip>
          </Box>
        )}
      </Box>
      {routeReason && !traceView && (
        <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', overflowWrap: 'anywhere', wordBreak: 'break-word' }}>
          Route reason: {compactExecutionText(routeReason, 480)}
        </Typography>
      )}
      {loading && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, px: 1, py: 0.75 }}>
          <CircularProgress size={14} />
          <Typography variant="caption" color="text.secondary">Loading run details...</Typography>
        </Box>
      )}
      {error && (
        <Typography variant="caption" color="error">
          {error}
        </Typography>
      )}
      {parallelSummary && (
        <Box sx={{ mx: 1, my: 0.75, p: 1, borderRadius: 1, border: 1, borderColor: parallelSummary.partial_evidence ? 'warning.main' : 'divider' }}>
          <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }}>
            Parallel dispatch{parallelSummary.partial_evidence ? ' · partial evidence' : ''}
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', overflowWrap: 'anywhere' }}>
            {Number(parallelSummary.completed || 0)}/{Number(parallelSummary.planned || 0)} completed
            {Number(parallelSummary.failed || 0) ? ` · ${parallelSummary.failed} failed` : ''}
            {Number(parallelSummary.timed_out || 0) ? ` · ${parallelSummary.timed_out} timed out` : ''}
            {Number(parallelSummary.retried || 0) ? ` · ${parallelSummary.retried} retries` : ''}
            {parallelSummary.elapsed_ms != null ? ` · ${Math.round(Number(parallelSummary.elapsed_ms))} ms worker time` : ''}
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.4, my: 0.5 }}>
            {Object.entries(PARALLEL_WORKER_STATUS_LABELS).map(([status, label]) => (
              Number(parallelSummary[status] || 0) > 0 ? <Chip key={status} size="small" variant="outlined" label={`${parallelSummary[status]} ${label}`} /> : null
            ))}
            <Chip size="small" variant="outlined" label={`barrier ${parallelSummary.barrier_state || 'pending'}`} />
            <Chip size="small" variant="outlined" label={`aggregation ${parallelSummary.aggregation_state || (parallelSummary.partial_evidence ? 'partial' : 'completed')}`} />
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
            Fan-out {Number(parallelSummary.fan_out_width ?? parallelSummary.planned ?? 0)}
            {' · '}peak {Number(parallelSummary.peak_concurrency || 0)}
            {parallelSummary.elapsed_ms != null ? ` · dispatch ${Math.round(Number(parallelSummary.elapsed_ms))} ms` : ''}
          </Typography>
          {(parallelSummary.evidence_packets_before_dedupe != null || parallelSummary.document_sources_before_dedupe != null || parallelSummary.web_sources_before_dedupe != null) && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
              Deduplication: evidence {Number(parallelSummary.evidence_packets_before_dedupe || 0)}→{Number(parallelSummary.evidence_packets_after_dedupe || 0)}
              {' · '}documents {Number(parallelSummary.document_sources_before_dedupe || 0)}→{Number(parallelSummary.document_sources_after_dedupe || 0)}
              {' · '}web {Number(parallelSummary.web_sources_before_dedupe || 0)}→{Number(parallelSummary.web_sources_after_dedupe || 0)}
            </Typography>
          )}
          {parallelSummary.dispatch_id && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', fontFamily: 'monospace', overflowWrap: 'anywhere' }}>
              Dispatch {parallelSummary.dispatch_id}
            </Typography>
          )}
          {parallelTasks.length > 0 && (
            <Box sx={{ mt: 0.5, display: 'flex', flexDirection: 'column', gap: 0.25 }}>
              {parallelTasks.map((task) => (
                <Box component="details" key={String(task.work_id)} sx={{ '& summary': { cursor: 'pointer' } }}>
                  <Typography component="summary" variant="caption" color="text.secondary">
                    {Number(task.ordinal || 0) + 1}. {task.worker_node_id || task.worker_type || 'worker'} · {task.status || 'queued'}
                    {Number(task.attempt || 0) > 1 ? ` · attempt ${task.attempt}` : ''}
                    {task.elapsed_ms != null ? ` · ${Math.round(Number(task.elapsed_ms))} ms` : ''}
                  </Typography>
                  {(Array.isArray(task.attempts) ? task.attempts : []).map((attempt: Record<string, any>) => (
                    <Typography key={`${task.work_id}:${attempt.attempt}`} variant="caption" color="text.secondary" sx={{ display: 'block', pl: 2 }}>
                      Attempt {Number(attempt.attempt || 1)} · {attempt.status || 'unknown'}
                      {attempt.reason ? ` · ${attempt.reason}` : ''}
                      {attempt.retryable === true ? ' · retryable' : attempt.retryable === false ? ' · non-retryable' : ''}
                      {attempt.elapsed_ms != null ? ` · ${Math.round(Number(attempt.elapsed_ms))} ms` : ''}
                      {attempt.occurred_at ? ` · ${attempt.occurred_at}` : ''}
                    </Typography>
                  ))}
                </Box>
              ))}
            </Box>
          )}
        </Box>
      )}
      {corrective && (
        <Box sx={{ mx: 1, my: 0.75, p: 1, borderRadius: 1, border: 1, borderColor: corrective.exhausted_budget_type ? 'warning.main' : 'divider' }}>
          <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }}>Corrective RAG</Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
            {Number(corrective.waves || 0)} corrective waves · {Number(corrective.distinct_work_items || 0)} work items · {Number(corrective.tool_attempts || 0)} attempts
            {Number(corrective.tool_retries || 0) ? ` · ${corrective.tool_retries} retries` : ''}
            {` · ${Number(corrective.partial_waves || 0)} partial waves`}
            {` · ${Number(corrective.successful_waves || 0)} successful`}
            {Number(corrective.failed_waves || 0) ? ` · ${corrective.failed_waves} failed` : ''}
            {Number(corrective.timed_out_waves || 0) ? ` · ${corrective.timed_out_waves} timed out` : ''}
            {Number(corrective.cancelled_waves || 0) ? ` · ${corrective.cancelled_waves} cancelled` : ''}
            {Number(corrective.source_expansions || 0) ? ` · ${corrective.source_expansions} source expansions` : ''}
            {Array.isArray(corrective.policy_filtered_memory_proposals) && corrective.policy_filtered_memory_proposals.length
              ? ` · ${corrective.policy_filtered_memory_proposals.length} memory proposals policy-filtered`
              : ''}
            {corrective.exhausted_budget_type ? ` · exhausted ${String(corrective.exhausted_budget_type).replaceAll('_', ' ')}` : ''}
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
            Packets {Number(corrective.accepted_packets || 0)} accepted / {Number(corrective.rejected_packets || 0)} rejected
            {' · '}support {Math.round(Number(corrective.support_ratio || 0) * 100)}%
            {' · '}{Number(corrective.citation_violations || 0)} citation violations
            {' · '}{Number(corrective.contradictions || 0)} contradictions
            {' · '}{Number(corrective.unresolved_gaps || 0)} gaps
            {groundingUsefulness ? ` · usefulness ${groundingUsefulness}` : ''}
            {corrective.termination_reason ? ` · stopped: ${String(corrective.termination_reason).replaceAll('_', ' ')}` : ''}
          </Typography>
          {(Array.isArray(corrective.history) ? corrective.history : []).map((wave: Record<string, any>) => (
            <Box component="details" key={`wave:${wave.wave_id}`} sx={{ mt: 0.4, '& summary': { cursor: 'pointer' } }}>
              <Typography component="summary" variant="caption">Wave {wave.wave_id} · {wave.reason || 'corrective retrieval'}</Typography>
              {(Array.isArray(wave.work_items) ? wave.work_items : []).map((item: Record<string, any>, index: number) => (
                <Typography key={`${wave.wave_id}:${index}`} variant="caption" color="text.secondary" sx={{ display: 'block', pl: 1.5, overflowWrap: 'anywhere' }}>
                  {item.worker_node_id || 'worker'}: {item.query || 'query'}{item.file_hash ? ` · document ${item.file_hash}` : ''}
                </Typography>
              ))}
            </Box>
          ))}
          {(Array.isArray(corrective.wave_outcomes) ? corrective.wave_outcomes : []).map((wave: Record<string, any>) => (
            <Box component="details" key={`outcome:${wave.wave_id}`} sx={{ mt: 0.4, '& summary': { cursor: 'pointer' } }}>
              <Typography component="summary" variant="caption">
                Wave {wave.wave_id} · {wave.outcome || wave.status || 'unknown'} · {Number(wave.completed || 0)}/{Number(wave.planned || 0)} workers completed
                {wave.partial ? ' · partial' : ''}{wave.latency_ms != null ? ` · ${Math.round(Number(wave.latency_ms))} ms` : ' · latency unavailable'}
              </Typography>
              {(Array.isArray(wave.work_items) ? wave.work_items : []).map((item: Record<string, any>, index: number) => (
                <Typography key={`${wave.wave_id}:outcome:${item.work_id || index}`} variant="caption" color="text.secondary" sx={{ display: 'block', pl: 1.5, overflowWrap: 'anywhere' }}>
                  {item.source_strategy || item.worker_node_id || 'worker'} · {item.status || 'unknown'} · {item.query || 'query'}
                  {item.query_id ? ` · query ${item.query_id}` : ''}{item.source_expansion ? ' · expanded source' : ''}
                </Typography>
              ))}
            </Box>
          ))}
          {retrievalQuality && (
            <Box component="details" sx={{ mt: 0.4, '& summary': { cursor: 'pointer' } }}>
              <Typography component="summary" variant="caption">Retrieval grade · {retrievalQuality.verdict || 'unknown'} · {Math.round(Number(retrievalQuality.confidence || 0) * 100)}%</Typography>
              {(Array.isArray(retrievalQuality.packet_assessments) ? retrievalQuality.packet_assessments : []).map((item: Record<string, any>) => (
                <Typography key={String(item.packet_id)} variant="caption" color="text.secondary" sx={{ display: 'block', pl: 1.5, overflowWrap: 'anywhere' }}>
                  {item.packet_id}: {item.eligible ? 'eligible' : 'rejected'} · {Math.round(Number(item.confidence || 0) * 100)}%
                  {(item.source_ids || []).length ? ` · ${(item.source_ids || []).join(', ')}` : ''}
                  {(item.coverage || []).length ? ` · covers ${(item.coverage || []).join(', ')}` : ''}
                  {(item.rejection_reasons || []).length ? ` · ${(item.rejection_reasons || []).join(', ').replaceAll('_', ' ')}` : ''}
                  {(item.instruction_injection_reasons || []).length ? ` · injection: ${(item.instruction_injection_reasons || []).join(', ').replaceAll('_', ' ')}` : ''}
                </Typography>
              ))}
            </Box>
          )}
          {grounding && (
            <Box component="details" sx={{ mt: 0.4, '& summary': { cursor: 'pointer' } }}>
              <Typography component="summary" variant="caption">
                {typeof grounding.grounded === 'boolean'
                  ? `Hermes grounding · ${grounding.grounded ? 'grounded' : 'evidence unavailable'} · ${Number(grounding.evidence_result_count || 0)} results`
                  : `Support and citations · ${Math.round(Number(grounding.supported_claim_ratio || 0) * 100)}% supported`}
              </Typography>
              {Array.isArray(grounding.successful_evidence_tools) && grounding.successful_evidence_tools.length > 0
                ? <Typography variant="caption" color="text.secondary" sx={{ display: 'block', pl: 1.5 }}>Tools: {grounding.successful_evidence_tools.join(', ')}</Typography>
                : null}
              {Array.isArray(grounding.failure_codes) && grounding.failure_codes.length > 0
                ? <Typography variant="caption" color="error" sx={{ display: 'block', pl: 1.5 }}>Failures: {grounding.failure_codes.join(', ').replaceAll('_', ' ')}</Typography>
                : null}
              {(Array.isArray(grounding.claims) ? grounding.claims : []).map((claim: Record<string, any>, index: number) => (
                <Typography key={`claim:${index}`} variant="caption" color="text.secondary" sx={{ display: 'block', pl: 1.5, overflowWrap: 'anywhere' }}>
                  {claim.claim_id ? `${claim.claim_id} · ` : ''}{claim.support}: {claim.claim}{(claim.source_ids || []).length ? ` · ${(claim.source_ids || []).join(', ')}` : ''}{claim.contradicted ? ' · contradicted' : ''}
                </Typography>
              ))}
              {(grounding.citation_violations || []).map((item: string, index: number) => <Typography key={`citation:${index}`} variant="caption" color="error" sx={{ display: 'block', pl: 1.5 }}>{item}</Typography>)}
              {(grounding.contradictions || []).map((item: Record<string, any>, index: number) => <Typography key={`contradiction:${index}`} variant="caption" color="error" sx={{ display: 'block', pl: 1.5 }}>{item.claim || 'Conflicting evidence'}{(item.claim_ids || []).length ? ` · claims ${(item.claim_ids || []).join(', ')}` : ''}{(item.source_ids || []).length ? ` · ${(item.source_ids || []).join(', ')}` : ''}</Typography>)}
              {(grounding.unresolved_gaps || []).map((item: string, index: number) => <Typography key={`gap:${index}`} variant="caption" color="warning.main" sx={{ display: 'block', pl: 1.5 }}>{item}</Typography>)}
            </Box>
          )}
        </Box>
      )}
      {pendingInterrupt && (
        <Box
          sx={{
            mx: 1,
            my: 0.75,
            p: 1,
            borderRadius: 1,
            bgcolor: 'rgba(25, 118, 210, 0.08)',
            display: 'flex',
            flexDirection: 'column',
            gap: 0.75,
          }}
        >
          <Typography variant="caption" sx={{ fontWeight: 600 }}>
            Human review: {interruptStatus}
          </Typography>
          {(pendingInterrupt.title || pendingInterrupt.prompt || pendingInterrupt.body) && (
            <Typography variant="caption" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
              {pendingInterrupt.title || pendingInterrupt.prompt || pendingInterrupt.body}
            </Typography>
          )}
          {interruptStatus === InterruptStatus.Pending && (
            <>
              {!responseOperation && (
                <Typography variant="caption" color="error">
                  This human-input request has an invalid runtime response contract.
                </Typography>
              )}
              {responseAvailability.visible && !responseAvailability.enabled && (
                <Typography variant="caption" color="text.secondary">
                  Human input is currently unavailable: {responseAvailability.disabledReason}
                </Typography>
              )}
              {isTaskOwnedRun ? (
                <Typography variant="caption" color="text.secondary">
                  Respond to this request in the Deep Research panel. Debug Trace is inspection-only for task runs.
                </Typography>
              ) : interruptOptions.length > 0 && (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.25 }}>
                  {interruptOptions.map((option) => (
                    <FormControlLabel
                      key={option.id}
                      sx={{ m: 0 }}
                      control={
                        <Checkbox
                          size="small"
                          checked={selectedOptionIds.includes(option.id)}
                          onChange={() => toggleOption(option.id)}
                          disabled={Boolean(resumeSubmitting)}
                        />
                      }
                      label={
                        <Typography variant="caption">
                          {option.label || option.id}
                        </Typography>
                      }
                    />
                  ))}
                </Box>
              )}
              {!isTaskOwnedRun && responseOperation && <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
                {renderInterruptAction(AgentRunResumeActionValue.Approve, 'Approve', <CheckIcon fontSize="inherit" />)}
                {renderInterruptAction(AgentRunResumeActionValue.ApproveForScope, 'Approve for this run', <CheckIcon fontSize="inherit" />)}
                {renderInterruptAction(AgentRunResumeActionValue.ApproveSelected, 'Approve selected', <CheckIcon fontSize="inherit" />)}
                {renderInterruptAction(AgentRunResumeActionValue.ContinueWithout, 'Continue', <PlayArrowIcon fontSize="inherit" />)}
                {renderInterruptAction(AgentRunResumeActionValue.Reject, 'Reject', <CloseIcon fontSize="inherit" />, 'error')}
              </Box>}
            </>
          )}
          {resumeMessage && (
            <Typography variant="caption" color="text.secondary">
              {resumeMessage}
            </Typography>
          )}
          {resumeError && (
            <Typography variant="caption" color="error">
              {resumeError}
            </Typography>
          )}
        </Box>
      )}
      {!loading && !error && runDetails && !debug && (
        <Typography variant="caption" color={retainedErrorMessage ? 'error' : 'text.secondary'} sx={{ px: 1, py: 0.75 }}>
          {unsupportedTraceFormat
            ? 'Trace format is no longer supported.'
            : retainedErrorMessage || (traceRefreshExhausted ? 'Trace not captured for this run.' : 'Retained execution trace is finalizing…')}
        </Typography>
      )}
      {debug && !traceView && (
        <Typography variant="caption" color="text.secondary" sx={{ px: 1, py: 0.75 }}>
          {unsupportedTraceFormat ? 'Trace format is no longer supported.' : 'Trace payload is incomplete.'}
        </Typography>
      )}
      {executionTraceView && (
        <>
      <AgentExecutionView
            runId={runId}
            threadId={executionThreadId}
            resolvedSpec={executionResolvedSpec}
        workflowId={executionWorkflowId}
        framework={executionFramework}
            traceView={executionTraceView}
            status={executionStatus}
            running={running}
            focusedTraceRefs={traceRefs}
            suspended={suspendHeavyContent}
            defaultFinalAnswerOpen={false}
            chatMode
            detailsAvailable={executionDetailsAvailable}
          />
        </>
      )}
    </Box>
  );
}

export default React.memo(AgentRunDebugPanel);
