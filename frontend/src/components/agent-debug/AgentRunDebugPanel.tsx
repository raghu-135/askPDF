import React, { useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DownloadIcon from '@mui/icons-material/Download';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { Box, Button, Checkbox, CircularProgress, FormControlLabel, IconButton, Tooltip, Typography } from '@mui/material';
import { resumeAgentRun, type AgentRunDetails, type AgentRunResumeAction, type AgentTraceRefs } from '../../lib/api';
import { AgentRunResumeAction as AgentRunResumeActionValue, AgentRunStatus, HitlSelectionMode, InterruptStatus } from '../../lib/enums';
import AgentRunHeaderChips from './AgentRunHeaderChips';
import { buildRunTraceView, buildTraceExportJson } from './agent-trace-projection';

const AgentDebugCanvas = dynamic(() => import('../agent-graph/AgentDebugCanvas'), { ssr: false });

export default function AgentRunDebugPanel({
  runId,
  routeReason,
  traceRefs,
  runDetails,
  loading,
  error,
  onRunDetailsChange,
}: {
  runId: string;
  routeReason?: string;
  traceRefs?: AgentTraceRefs | null;
  runDetails?: AgentRunDetails;
  loading?: boolean;
  error?: string;
  onRunDetailsChange?: (runDetails: AgentRunDetails) => void;
}) {
  const [copyStatus, setCopyStatus] = useState<'idle' | 'copied' | 'failed'>('idle');
  const [resumeSubmitting, setResumeSubmitting] = useState<AgentRunResumeAction | null>(null);
  const [resumeError, setResumeError] = useState<string | null>(null);
  const [resumeMessage, setResumeMessage] = useState<string | null>(null);
  const [selectedOptionIds, setSelectedOptionIds] = useState<string[]>([]);
  const resumeSubmissionKeyRef = useRef<string | null>(null);
  const debug = runDetails?.debug;
  const pendingInterrupt = runDetails?.pending_interrupt;
  const interruptStatus = pendingInterrupt?.status || (pendingInterrupt ? InterruptStatus.Pending : undefined);
  const allowedActions = Array.isArray(pendingInterrupt?.allowed_actions)
    ? pendingInterrupt.allowed_actions.map(String)
    : [];
  const traceView = runDetails ? buildRunTraceView(runDetails) : undefined;
  const trace = traceView?.trace;
  const traceJson = buildTraceExportJson(traceView);
  const interruptOptions = Array.isArray(pendingInterrupt?.options)
    ? pendingInterrupt.options.filter((option) => option && typeof option.id === 'string')
    : [];
  const selectionMode = String(pendingInterrupt?.selection_mode || HitlSelectionMode.Single);
  const isMultiSelect = selectionMode === HitlSelectionMode.Multi || selectionMode === HitlSelectionMode.SingleOrMulti;

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
    const submissionKey = `${runId}:${pendingInterrupt.interrupt_id}:${pendingInterrupt.resume_version ?? 1}:${action}`;
    if (resumeSubmissionKeyRef.current) return;
    resumeSubmissionKeyRef.current = submissionKey;
    setResumeSubmitting(action);
    setResumeError(null);
    setResumeMessage(null);
    try {
      const response = await resumeAgentRun(runId, {
        action,
        interrupt_id: pendingInterrupt.interrupt_id,
        resume_token: pendingInterrupt.resume_token || undefined,
        resume_version: pendingInterrupt.resume_version || undefined,
        thread_id: runDetails.thread_id,
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
    if (!allowedActions.includes(action)) return null;
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
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75 }}>
      <Typography variant="caption" sx={{ display: 'block', wordBreak: 'break-all' }}>
        Run ID: {runId}
      </Typography>
      {routeReason && (
        <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
          Route reason: {routeReason}
        </Typography>
      )}
      {loading && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <CircularProgress size={14} />
          <Typography variant="caption" color="text.secondary">Loading run details...</Typography>
        </Box>
      )}
      {error && (
        <Typography variant="caption" color="error">
          {error}
        </Typography>
      )}
      {pendingInterrupt && (
        <Box
          sx={{
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
              {interruptOptions.length > 0 && (
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
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
                {renderInterruptAction(AgentRunResumeActionValue.Approve, 'Approve', <CheckIcon fontSize="inherit" />)}
                {renderInterruptAction(AgentRunResumeActionValue.ApproveSelected, 'Approve selected', <CheckIcon fontSize="inherit" />)}
                {renderInterruptAction(AgentRunResumeActionValue.ContinueWithout, 'Continue', <PlayArrowIcon fontSize="inherit" />)}
                {renderInterruptAction(AgentRunResumeActionValue.Reject, 'Reject', <CloseIcon fontSize="inherit" />, 'error')}
              </Box>
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
        <Typography variant="caption" color="text.secondary">
          Trace not captured for this run.
        </Typography>
      )}
      {debug && !traceView && (
        <Typography variant="caption" color="text.secondary">
          Trace payload is incomplete.
        </Typography>
      )}
      {debug && runDetails && traceView && (
        <>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, alignItems: 'center' }}>
            <AgentRunHeaderChips runDetails={runDetails} traceView={traceView} />
            {trace && (
              <>
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
              </>
            )}
          </Box>
          <AgentDebugCanvas
            resolvedSpec={runDetails.resolved_spec_json}
            workflowId={runDetails.workflow_id}
            traceView={traceView}
            focusedTraceRefs={traceRefs}
          />
        </>
      )}
    </Box>
  );
}
