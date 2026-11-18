import React, { useEffect, useMemo, useRef, useState } from 'react';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DownloadIcon from '@mui/icons-material/Download';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import { Box, Button, Chip, CircularProgress, IconButton, Tooltip, Typography } from '@mui/material';
import { getAgentRun, resumeAgentRun, type AgentRunDetails, type AgentRunResumeAction, type AgentTraceRefs } from '../../lib/api';
import { isRuntimeOperationEnabled, runtimeInterruptResponseOperation, runtimeOperationAvailability } from '../../lib/runtime-capabilities';
import { AgentRunResumeAction as AgentRunResumeActionValue, AgentRunStatus, InterruptStatus } from '../../lib/enums';
import { buildCorrectiveInspection, buildRunTraceView, buildTraceExportJson, getRetainedRunErrorMessage, getRunTrace, mergeLiveAndRetainedTraceViews, parseRunDebug, shouldRefreshRetainedTrace, type TraceRunView } from './agent-trace-projection';
import AgentExecutionView from '../agent-graph/AgentExecutionView';
import { compactExecutionText } from '../agent-graph/agent-execution-display';
import { isTaskOwnedAgentRun } from '../../lib/deep-research-ui-state';
import { useAgentRunCapabilities } from '../../lib/use-agent-run-capabilities';
import { isValidTraceId } from '../../lib/trace-tabs';

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
  onOpenDelegatedRun,
}: {
  runId?: string;
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
  onOpenDelegatedRun?: (childRunId: string, childWorkflowId?: string) => void;
}) {
  const normalizedRunId = isValidTraceId(runId) ? runId.trim() : '';
  const [copyStatus, setCopyStatus] = useState<'idle' | 'copied' | 'failed'>('idle');
  const [resumeSubmitting, setResumeSubmitting] = useState<AgentRunResumeAction | null>(null);
  const [resumeError, setResumeError] = useState<string | null>(null);
  const [resumeMessage, setResumeMessage] = useState<string | null>(null);
  const resumeSubmissionKeyRef = useRef<string | null>(null);
  const traceRefreshAttemptedRef = useRef(new Map<string, number>());
  const [traceRefreshExhausted, setTraceRefreshExhausted] = useState(false);
  const [refreshedRunDetails, setRefreshedRunDetails] = useState<AgentRunDetails | undefined>();
  const runDetails = refreshedRunDetails?.id === normalizedRunId ? refreshedRunDetails : providedRunDetails;
  const runCapabilities = useAgentRunCapabilities(
    normalizedRunId || null,
    runDetails?.thread_id || threadId,
    `${runDetails?.status}:${runDetails?.runtime_binding_status}:${runDetails?.pending_interrupt?.interrupt_id}:${runDetails?.pending_interrupt?.status}:${runDetails?.pending_interrupt?.resume_version}`,
  ).capabilities;
  const debug = runDetails?.debug;
  const retainedErrorMessage = runDetails ? getRetainedRunErrorMessage(runDetails) : null;
  const pendingInterrupt = runDetails?.pending_interrupt;
  const isTaskOwnedRun = isTaskOwnedAgentRun(runDetails);
  const interruptStatus = pendingInterrupt?.status;
  const runTerminal = ['completed', 'failed', 'cancelled', 'expired', 'recovery_required'].includes(
    String(runDetails?.status || ''),
  );
  const interruptAwaitingResponse = interruptStatus === InterruptStatus.Pending
    && !pendingInterrupt?.decision
    && !runTerminal;
  const showPendingInterruptBanner = Boolean(pendingInterrupt && interruptAwaitingResponse);
  const responseOperation = runtimeInterruptResponseOperation(pendingInterrupt);
  const responseAvailability = responseOperation
    ? runtimeOperationAvailability(runCapabilities, responseOperation)
    : { visible: false, enabled: false, disabledReason: 'invalid_interrupt_response_operation' };
  const traceView = useMemo(() => runDetails ? buildRunTraceView(runDetails) : undefined, [runDetails]);
  const debugParseResult = useMemo(() => runDetails?.debug && runDetails ? parseRunDebug(runDetails) : null, [runDetails]);
  const debugParseError = debugParseResult && 'reason' in debugParseResult ? debugParseResult.reason : null;
  const debugParseCorrelationId = debugParseResult && 'correlationId' in debugParseResult ? debugParseResult.correlationId : null;
  const liveParseError = liveTraceView?.parseError;
  const executionTraceView = useMemo(
    () => liveTraceView?.parseError ? liveTraceView : liveTraceView ? mergeLiveAndRetainedTraceViews(liveTraceView, traceView) : traceView,
    [liveTraceView, traceView],
  );
  const traceJson = useMemo(() => buildTraceExportJson(executionTraceView), [executionTraceView]);
  const executionThreadId = runDetails?.thread_id || null;
  const executionWorkflowId = runDetails?.workflow_id;
  const executionFramework = (runDetails as any)?.framework || (runDetails as any)?.runtime_metadata?.framework;
  const executionResolvedSpec = runDetails?.resolved_spec_json;
  const executionStatus = runDetails?.status || (running ? 'running' : undefined);
  const executionDetailsAvailable = Boolean(runDetails && !running);
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
  const delegationMeta = useMemo(() => {
    if (!runDetails) return null;
    const technical = (
      (debug?.technical_details && typeof debug.technical_details === 'object' && !Array.isArray(debug.technical_details))
        ? debug.technical_details
        : getRunTrace(runDetails)?.technical_details
    ) as Record<string, unknown> | undefined;
    if (!technical) return null;
    const childRunId = String(technical.child_run_id || technical.delegate_child_run_id || '').trim();
    const childWorkflowId = String(technical.child_workflow_id || technical.delegate_child_workflow_id || '').trim();
    return childRunId ? { childRunId, childWorkflowId: childWorkflowId || undefined } : null;
  }, [debug?.technical_details, runDetails]);

  useEffect(() => {
    setRefreshedRunDetails(undefined);
    setTraceRefreshExhausted(false);
  }, [normalizedRunId, providedRunDetails]);

  useEffect(() => {
    const metadataThreadId = threadId || runDetails?.thread_id;
    if (runDetails || !metadataThreadId || !normalizedRunId) return undefined;
    let active = true;
    void getAgentRun(normalizedRunId, metadataThreadId)
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
  }, [normalizedRunId, onRunDetailsChange, runDetails, threadId]);

  useEffect(() => {
    if (!normalizedRunId || !runDetails || !executionThreadId || !shouldRefreshRetainedTrace(runDetails)) return;
    const live = String(runDetails.status) === 'running' || String(runDetails.status) === 'awaiting_human';
    if (!live) {
      const attempts = traceRefreshAttemptedRef.current.get(normalizedRunId) || 0;
      if (attempts >= 5) {
        setTraceRefreshExhausted(true);
        return;
      }
      traceRefreshAttemptedRef.current.set(normalizedRunId, attempts + 1);
    }
    const attempts = traceRefreshAttemptedRef.current.get(normalizedRunId) || 1;
    const timer = window.setTimeout(() => {
      void getAgentRun(normalizedRunId, executionThreadId)
        .then((refreshed) => {
          setRefreshedRunDetails(refreshed);
          onRunDetailsChange?.(refreshed);
        })
        .catch(() => undefined);
    }, live ? 1000 : 500 * attempts);
    return () => window.clearTimeout(timer);
  }, [executionThreadId, normalizedRunId, onRunDetailsChange, runDetails]);

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
    link.download = `agent-trace-${normalizedRunId || 'pending'}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleResume = async (action: AgentRunResumeAction, selectedOptionIds?: string[]) => {
    if (!runDetails || !pendingInterrupt?.interrupt_id) return;
    if (!responseOperation) {
      setResumeError('This human-input request has an invalid runtime response contract.');
      return;
    }
    const submissionKey = `${normalizedRunId}:${pendingInterrupt.interrupt_id}:${pendingInterrupt.resume_version ?? 1}:${action}`;
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
      const response = await resumeAgentRun(normalizedRunId, {
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

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', width: '100%', minWidth: 0, maxWidth: '100%', overflowX: 'hidden' }}>
      <Box sx={{ px: 1, py: 0.5, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 0.5, bgcolor: 'background.paper', borderBottom: 1, borderColor: 'divider' }}>
        <Tooltip title={normalizedRunId || 'Run ID pending'} arrow>
          <Typography variant="caption" color="text.secondary">
            {normalizedRunId ? `Run …${normalizedRunId.slice(-8)}` : 'Run pending'}
          </Typography>
        </Tooltip>
        {traceJson && (
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
      {delegationMeta && onOpenDelegatedRun && (
        <Box sx={{ px: 1, pb: 0.5 }}>
          <Button
            size="small"
            variant="outlined"
            startIcon={<OpenInNewIcon />}
            data-testid="debug-delegated-child-run-link"
            onClick={() => onOpenDelegatedRun(delegationMeta.childRunId, delegationMeta.childWorkflowId)}
          >
            Open delegated child run{delegationMeta.childWorkflowId ? ` (${delegationMeta.childWorkflowId})` : ''}
          </Button>
        </Box>
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
      {showPendingInterruptBanner && pendingInterrupt && (
        <Box
          data-testid="debug-human-review-banner"
          sx={{
            mx: 1,
            my: 0.75,
            display: 'flex',
            flexDirection: 'column',
            gap: 0.75,
          }}
        >
          <Typography variant="caption" sx={{ fontWeight: 600, px: 0.5 }}>
            Human review: {interruptStatus}
          </Typography>
          {interruptStatus === InterruptStatus.Pending && isTaskOwnedRun ? (
            <Typography variant="caption" color="text.secondary" sx={{ px: 0.5 }}>
              Respond to this request in the Deep Research panel. Debug Trace is inspection-only for task runs.
            </Typography>
          ) : interruptStatus === InterruptStatus.Pending && !responseOperation ? (
            <Typography variant="caption" color="error" sx={{ px: 0.5 }}>
              This human-input request has an invalid runtime response contract.
            </Typography>
          ) : interruptStatus === InterruptStatus.Pending ? (
            <Typography variant="caption" color="text.secondary" sx={{ px: 0.5 }}>
              {isTaskOwnedRun
                ? 'Approval is pending in the Deep Research panel. Debug Trace records the interrupt event only.'
                : 'Approval is pending in chat. Use the approval card in the conversation to continue; this trace shows the interrupt event only.'}
            </Typography>
          ) : null}
          {resumeMessage && (
            <Typography variant="caption" color="text.secondary" sx={{ px: 0.5 }}>
              {resumeMessage}
            </Typography>
          )}
        </Box>
      )}
      {!loading && !error && runDetails && !debug && (
        <Typography variant="caption" color={retainedErrorMessage ? 'error' : 'text.secondary'} sx={{ px: 1, py: 0.75 }}>
          {retainedErrorMessage || (traceRefreshExhausted ? 'Trace not captured for this run.' : 'Retained execution trace is finalizing…')}
        </Typography>
      )}
      {debug && !traceView && (
        <Box sx={{ px: 1, py: 0.75 }}>
          <Typography variant="caption" color="error" sx={{ display: 'block' }}>Debug trace data could not be parsed.</Typography>
          {debugParseError && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', overflowWrap: 'anywhere' }}>{debugParseError}</Typography>
          )}
          {debugParseCorrelationId && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>Correlation ID: {debugParseCorrelationId}</Typography>
          )}
        </Box>
      )}
      {liveParseError && (
        <Box sx={{ px: 1, py: 0.75 }}>
          <Typography variant="caption" color="error" sx={{ display: 'block' }}>Debug trace data could not be parsed.</Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', overflowWrap: 'anywhere' }}>{liveParseError}</Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>Correlation ID: {liveTraceView?.parseCorrelationId || `trace:${normalizedRunId || 'pending'}`}</Typography>
        </Box>
      )}
      {executionTraceView && (!liveParseError || executionTraceView.events.length > 0 || executionTraceView.operations.length > 0) && (
        <>
      <AgentExecutionView
            runId={normalizedRunId}
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
            onOpenDelegatedRun={onOpenDelegatedRun}
          />
        </>
      )}
    </Box>
  );
}

export default React.memo(AgentRunDebugPanel);
