import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import AccountTreeOutlinedIcon from '@mui/icons-material/AccountTreeOutlined';
import BuildOutlinedIcon from '@mui/icons-material/BuildOutlined';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import MemoryIcon from '@mui/icons-material/Memory';
import TimerOutlinedIcon from '@mui/icons-material/TimerOutlined';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import { Accordion, AccordionDetails, AccordionSummary, Alert, Box, Chip, CircularProgress, IconButton, Paper, Stack, Tooltip, Typography } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { getAgentRunOperationDetails, type AgentRunOperationDetail } from '../../lib/api';
import type { TraceOperationView, TraceRunView } from '../agent-debug/agent-trace-projection';
import type { AgentGraphSelection, AgentNodeVisitRef, AgentTraceRefs } from './agent-graph-types';
import AgentNodeExecutionDetails from './AgentNodeExecutionDetails';
import AgentExecutionStatusIcon from './AgentExecutionStatusIcon';
import { formatDurationMs } from '../../lib/formatDuration';
import { TraceLlmUsageTooltip, TraceOperationsTooltip, TraceToolsTooltip } from '../agent-debug/AgentRunTraceTooltips';
import GenericTraceTimeline from '../agent-debug/GenericTraceTimeline';
import TraceDiagnosticsPanel from '../agent-debug/TraceDiagnosticsPanel';
import TraceVisualizationSlot from '../agent-debug/TraceVisualizationSlot';
import { compactExecutionText } from './agent-execution-display';
import {
  agentNodeVisitKey,
  getChronologicalNodeVisits,
  getNextNodeVisit,
  getNodeVisitRoute,
  getPreviousNodeVisit,
  toAgentNodeVisitRef,
} from './agent-node-visits';

const visitKey = (operation: Pick<TraceOperationView, 'id' | 'visitIndex'>) => agentNodeVisitKey(operation);

const nodeSummary = (node: TraceOperationView) => {
  const raw = node.raw || {};
  const detail = raw.detail || {};
  const event = detail.event || raw;
  let summary: unknown;
  if (node.status === 'error') summary = `Failed: ${detail.error?.raw_message || event.error?.raw_message || node.error?.raw_message || 'node execution failed'}`;
  else if (node.skipped) summary = `Skipped${event.skip_reason ? `: ${event.skip_reason}` : ''}`;
  else if (event.evaluator_route) summary = `Evaluated evidence and chose ${event.evaluator_route}.`;
  else if (event.route) summary = `Selected the ${event.route} route${event.route_reason ? `: ${event.route_reason}` : '.'}`;
  else if (Array.isArray(event.execution_plan)) summary = `Planned ${event.execution_plan.length} step${event.execution_plan.length === 1 ? '' : 's'}: ${event.execution_plan.join(' → ')}.`;
  else if (node.usedMemoryIdCount) summary = `Recalled ${node.usedMemoryIdCount} long-term memor${Number(node.usedMemoryIdCount) === 1 ? 'y' : 'ies'}.`;
  else if (event.document_source_count || event.web_source_count) summary = `Retrieved ${Number(event.document_source_count || 0) + Number(event.web_source_count || 0)} source${Number(event.document_source_count || 0) + Number(event.web_source_count || 0) === 1 ? '' : 's'}.`;
  else if (event.answer_chars) summary = `Generated an answer (${event.answer_chars} characters).`;
  else summary = node.status === 'active' ? 'Running…' : 'Completed this step.';
  return compactExecutionText(summary, 260);
};

const formatTokenCount = (value: unknown) => {
  const count = Number(value);
  if (!Number.isFinite(count) || count <= 0) return undefined;
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(count >= 10_000_000 ? 0 : 1)}m`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(count >= 10_000 ? 0 : 1)}k`;
  return count.toLocaleString();
};

function AgentExecutionView({
  runId,
  threadId,
  resolvedSpec,
  framework,
  workflowId,
  traceView,
  status,
  running = false,
  focusedTraceRefs,
  defaultFinalAnswerOpen = false,
  suspended = false,
  chatMode = false,
  detailsAvailable = true,
}: {
  runId?: string | null;
  threadId?: string | null;
  resolvedSpec?: Record<string, any>;
  framework?: string;
  workflowId?: string;
  traceView: TraceRunView;
  status?: string;
  running?: boolean;
  focusedTraceRefs?: AgentTraceRefs | null;
  defaultFinalAnswerOpen?: boolean;
  suspended?: boolean;
  chatMode?: boolean;
  detailsAvailable?: boolean;
}) {
  const initialDetails = useMemo(() => {
    const result: Record<string, AgentRunOperationDetail> = {};
    traceView.operations.forEach((node) => {
      if (node.raw?.detail) result[visitKey(node)] = node.raw.detail as AgentRunOperationDetail;
    });
    return result;
  }, [traceView.operations]);
  const [details, setDetails] = useState<Record<string, AgentRunOperationDetail>>(initialDetails);
  const [expanded, setExpanded] = useState<string | false>(false);
  const [selectedVisit, setSelectedVisit] = useState<AgentNodeVisitRef | null>(null);
  const [selectedTopologyNodeId, setSelectedTopologyNodeId] = useState<string | null>(null);
  const [loadingKey, setLoadingKey] = useState<string | null>(null);
  const [detailErrors, setDetailErrors] = useState<Record<string, string>>({});
  const [revealRequest, setRevealRequest] = useState<{ key: string; token: number } | null>(null);
  const [progressOpen, setProgressOpen] = useState(true);
  const [finalAnswerOpen, setFinalAnswerOpen] = useState(defaultFinalAnswerOpen);
  const [focusedEventId, setFocusedEventId] = useState<string | null>(null);
  const inFlightDetailKeys = useRef(new Set<string>());
  const timelineRows = useRef(new Map<string, HTMLElement>());
  const timelineSummaries = useRef(new Map<string, HTMLElement>());
  const detailContextKey = `${runId || 'live'}:${threadId || ''}`;
  const detailContextRef = useRef(detailContextKey);
  const selectionContextRef = useRef(detailContextKey);

  useEffect(() => {
    if (detailContextRef.current !== detailContextKey) {
      detailContextRef.current = detailContextKey;
      selectionContextRef.current = '';
      inFlightDetailKeys.current.clear();
      setDetails(initialDetails);
      setExpanded(false);
      setSelectedVisit(null);
      setSelectedTopologyNodeId(null);
      setRevealRequest(null);
      setLoadingKey(null);
      setDetailErrors({});
      setProgressOpen(true);
      setFinalAnswerOpen(defaultFinalAnswerOpen);
      setFocusedEventId(null);
      return;
    }
    setDetails((current) => {
      const changed = Object.entries(initialDetails).some(([key, detail]) => current[key] !== detail);
      return changed ? { ...current, ...initialDetails } : current;
    });
  }, [defaultFinalAnswerOpen, detailContextKey, initialDetails]);

  const loadDetail = useCallback(async (node: TraceOperationView) => {
    const key = visitKey(node);
    const requestKey = `${detailContextKey}:${key}`;
    if (!detailsAvailable || details[key] || inFlightDetailKeys.current.has(requestKey) || !runId || !threadId || node.status === 'active') return;
    inFlightDetailKeys.current.add(requestKey);
    setLoadingKey(key);
    setDetailErrors((current) => ({ ...current, [key]: '' }));
    try {
      const detail = await getAgentRunOperationDetails(runId, threadId, node.id, node.visitIndex || 1);
      if (detailContextRef.current === detailContextKey) {
        setDetails((current) => ({ ...current, [key]: detail }));
      }
    } catch (error: any) {
      if (detailContextRef.current === detailContextKey) {
        setDetailErrors((current) => ({ ...current, [key]: error?.message || 'Full details are unavailable.' }));
      }
    } finally {
      inFlightDetailKeys.current.delete(requestKey);
      if (detailContextRef.current === detailContextKey) {
        setLoadingKey((current) => current === key ? null : current);
      }
    }
  }, [detailContextKey, details, detailsAvailable, runId, threadId]);

  const selectVisit = useCallback((node: TraceOperationView, open: boolean) => {
    const key = visitKey(node);
    selectionContextRef.current = detailContextKey;
    setExpanded(open ? key : false);
    setSelectedVisit(toAgentNodeVisitRef(node));
    setSelectedTopologyNodeId(String(node.topologyRef?.id || node.id));
    if (open) {
      void loadDetail(node);
    }
  }, [detailContextKey, loadDetail]);

  const revealVisit = useCallback((visit: AgentNodeVisitRef) => {
    const node = traceView.operations.find((row) => visitKey(row) === agentNodeVisitKey(visit));
    if (!node) return;
    selectVisit(node, true);
    setProgressOpen(true);
    setRevealRequest((current) => ({ key: agentNodeVisitKey(visit), token: (current?.token || 0) + 1 }));
  }, [selectVisit, traceView.operations]);

  useEffect(() => {
    if (!revealRequest || expanded !== revealRequest.key) return;
    // Wait until the accordion and its details have committed before measuring the row.
    const firstFrame = window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        timelineRows.current.get(revealRequest.key)?.scrollIntoView({ behavior: 'smooth', block: 'start', inline: 'nearest' });
        timelineSummaries.current.get(revealRequest.key)?.focus({ preventScroll: true });
      });
    });
    return () => window.cancelAnimationFrame(firstFrame);
  }, [expanded, revealRequest]);

  const handleGraphSelection = useCallback((selection: AgentGraphSelection) => {
    if (!selection || selection.kind !== 'node') {
      setSelectedVisit(null);
      setSelectedTopologyNodeId(null);
      return;
    }
    setSelectedTopologyNodeId(selection.node.id);
    const node = [...traceView.operations].reverse().find((row) => (
      String(row.topologyRef?.id || row.id) === selection.node.id
    ));
    if (node) revealVisit(toAgentNodeVisitRef(node));
    else setSelectedVisit(null);
  }, [revealVisit, traceView.operations]);

  useEffect(() => {
    if (!selectedVisit) return;
    if (selectionContextRef.current !== detailContextKey) return;
    const selectedKey = agentNodeVisitKey(selectedVisit);
    if (traceView.operations.some((node) => visitKey(node) === selectedKey)) return;
    const fallback = [...traceView.operations].reverse().find((node) => node.id === selectedVisit.nodeId);
    if (fallback) {
      setSelectedVisit(toAgentNodeVisitRef(fallback));
    } else {
      setSelectedVisit(null);
    }
  }, [detailContextKey, selectedVisit, traceView.operations]);

  const finalOutput = traceView.finalOutput;
  const memoryDebug = traceView.memory;
  const runDuration = formatDurationMs(Number(traceView.metrics.duration_ms));
  const tokenCount = formatTokenCount(traceView.metrics.llm_token_count_total);
  const copyAnswer = useCallback(async () => {
    if (finalOutput?.answer && navigator.clipboard) await navigator.clipboard.writeText(finalOutput.answer);
  }, [finalOutput?.answer]);

  if (suspended) {
    return (
      <Paper variant="outlined" sx={{ width: '100%', minWidth: 0, p: 0.8 }}>
        <Stack direction="row" spacing={0.75} alignItems="center">
          <CircularProgress size={14} />
          <Typography variant="caption" color="text.secondary">Trace preview resumes after resizing.</Typography>
        </Stack>
      </Paper>
    );
  }

  return (
    <Stack spacing={0} sx={{ width: '100%', minWidth: 0, maxWidth: '100%', overflowX: 'hidden' }}>
      <Paper elevation={0} square sx={{ width: '100%', minWidth: 0, maxWidth: '100%', overflowX: 'hidden', p: 1, bgcolor: 'background.default' }}>
        <Box component="details" open={progressOpen} onToggle={(event) => setProgressOpen(event.currentTarget.open)}>
          <Box component="summary" sx={{ cursor: 'pointer', listStylePosition: 'inside', '&::marker': { fontSize: '0.78rem' } }}>
            <Stack direction="row" spacing={0.6} alignItems="center" sx={{ display: 'inline-flex', ml: 0.5, mb: progressOpen ? 0.75 : 0, flexWrap: 'wrap', rowGap: 0.45, verticalAlign: 'middle' }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 700, mr: 0.25 }}>{running ? 'Live progress' : 'Execution progress'}</Typography>
              <AgentExecutionStatusIcon status={status || (running ? 'running' : 'completed')} size={17} />
              {traceView.route && (
                <Chip size="small" variant="outlined" label={traceView.route} aria-label={`Route: ${traceView.route}`} sx={{ height: 22 }} />
              )}
              {runDuration && <Chip size="small" variant="outlined" icon={<TimerOutlinedIcon />} label={runDuration} sx={{ height: 22 }} />}
              <Tooltip title={<TraceOperationsTooltip operations={traceView.operations} usedCount={traceView.usedOperationCount} availableCount={traceView.availableOperationCount} />} arrow>
                <Chip aria-label={`${traceView.usedOperationCount} operations, ${traceView.operations.length} visits`} size="small" variant="outlined" icon={<AccountTreeOutlinedIcon />} label={`${traceView.usedOperationCount}o${traceView.operations.length !== traceView.usedOperationCount ? ` · ${traceView.operations.length}v` : ''}`} sx={{ height: 22 }} />
              </Tooltip>
              {traceView.tools.length > 0 && (
                <Tooltip title={<TraceToolsTooltip tools={traceView.tools} />} arrow>
                  <Chip aria-label={`${traceView.tools.length} tool calls`} size="small" variant="outlined" icon={<BuildOutlinedIcon />} label={traceView.tools.length} sx={{ height: 22 }} />
                </Tooltip>
              )}
              {tokenCount && (
                <Tooltip title={<TraceLlmUsageTooltip metrics={traceView.metrics} />} arrow>
                  <Chip size="small" variant="outlined" label={`${tokenCount} tokens`} sx={{ height: 22 }} />
                </Tooltip>
              )}
              {traceView.warningCount > 0 && (
                <Tooltip title={`${traceView.warningCount} warnings`} arrow>
                  <Chip size="small" color="warning" variant="outlined" icon={<WarningAmberIcon />} label={traceView.warningCount} sx={{ height: 22 }} />
                </Tooltip>
              )}
              {traceView.errorCount > 0 && (
                <Tooltip title={`${traceView.errorCount} errors`} arrow>
                  <Chip size="small" color="error" variant="outlined" icon={<ErrorOutlineIcon />} label={traceView.errorCount} sx={{ height: 22 }} />
                </Tooltip>
              )}
            </Stack>
          </Box>
          {progressOpen && (
            <>
        {memoryDebug && (memoryDebug.recalledCount > 0 || memoryDebug.searchedScopes.length > 0) && (
          <Paper variant="outlined" sx={{ p: 0.75, mb: 0.75, bgcolor: 'background.default' }}>
            <Stack direction="row" spacing={0.6} alignItems="center" sx={{ flexWrap: 'wrap', rowGap: 0.45 }}>
              <MemoryIcon sx={{ fontSize: 16 }} color="primary" />
              <Typography variant="caption" sx={{ fontWeight: 700 }}>Memory debug</Typography>
              {memoryDebug.searchedScopes.length > 0 && (
                <Tooltip title={memoryDebug.searchedScopes.map(scope => `${scope.scope_type}:${scope.scope_id}`).join('\n')} arrow>
                  <Chip size="small" variant="outlined" label={`${memoryDebug.searchedScopes.length} scopes`} sx={{ height: 22 }} />
                </Tooltip>
              )}
              {memoryDebug.recalledCount > 0 && (
                <Tooltip title={memoryDebug.recalledMemoryIds.join('\n')} arrow>
                  <Chip size="small" variant="outlined" label={`${memoryDebug.recalledCount} recalled`} sx={{ height: 22 }} />
                </Tooltip>
              )}
            </Stack>
          </Paper>
        )}
        <TraceDiagnosticsPanel
          diagnostics={traceView.diagnostics}
          onShowEvent={setFocusedEventId}
          onOpenOperation={(operationId, attempt) => {
            const operation = traceView.operations.find((row) => row.id === operationId && (!attempt || row.visitIndex === attempt))
              || [...traceView.operations].reverse().find((row) => row.id === operationId);
            if (operation) revealVisit(toAgentNodeVisitRef(operation));
          }}
        />
        {traceView.operations.length === 0 ? (
          <Typography variant="body2" color="text.secondary">Start a run to see each operation.</Typography>
        ) : traceView.operations.map((node, index) => {
          const key = visitKey(node);
          const detail = details[key];
          const nodeVisits = getChronologicalNodeVisits(traceView.operations, node.id);
          const visitPosition = nodeVisits.findIndex((visit) => visitKey(visit) === key);
          const visitRef = toAgentNodeVisitRef(node);
          const previousVisit = getPreviousNodeVisit(traceView.operations, visitRef);
          const nextVisit = getNextNodeVisit(traceView.operations, visitRef);
          const route = getNodeVisitRoute(node);
          const summary = nodeSummary(node);
          const formattedDuration = formatDurationMs(node.durationMs);
          const routeReason = compactExecutionText(node.routeReason, 480);
          const hasNodeError = Boolean(node.error && Object.keys(node.error).length > 0);
          const visitTools = traceView.tools.filter((tool) => (
            tool.callerNode === node.id && Number(tool.callerVisitIndex || 1) === visitRef.visitIndex
          ));
          const visitModels = traceView.models.filter((model) => (
            model.operation_id === node.id && Number(model.visit_index || 1) === visitRef.visitIndex
          ));
          const activeModel = [...visitModels].reverse().find((model) => model.status === 'started' || model.status === 'active');
          return (
            <Accordion
              key={`${key}:${index}`}
              ref={(element) => {
                if (element) timelineRows.current.set(key, element);
                else timelineRows.current.delete(key);
              }}
              expanded={expanded === key}
              onChange={(_, open) => selectVisit(node, open)}
              disableGutters
              sx={{
                width: '100%',
                ml: node.parentOperationId ? 1.5 : 0,
                minWidth: 0,
                maxWidth: '100%',
                overflow: 'hidden',
                scrollMarginTop: 16,
                contentVisibility: 'auto',
                contain: 'layout paint style',
                containIntrinsicSize: '38px',
                ...(selectedVisit && agentNodeVisitKey(selectedVisit) === key ? { borderColor: 'primary.main' } : {}),
              }}
            >
              <AccordionSummary
                ref={(element) => {
                  if (element) timelineSummaries.current.set(key, element);
                  else timelineSummaries.current.delete(key);
                }}
                tabIndex={-1}
                expandIcon={<ExpandMoreIcon titleAccess="Expand or collapse invocation details" sx={{ fontSize: 19 }} />}
                sx={{
                  width: '100%',
                  minWidth: 0,
                  maxWidth: '100%',
                  overflow: 'hidden',
                  minHeight: 38,
                  px: 0.75,
                  '&.Mui-expanded': { minHeight: 38 },
                  '& .MuiAccordionSummary-content': { minWidth: 0, maxWidth: 'calc(100% - 28px)', overflow: 'hidden', my: 0.45 },
                  '& .MuiAccordionSummary-content.Mui-expanded': { my: 0.45 },
                  '& .MuiAccordionSummary-expandIconWrapper': { flexShrink: 0 },
                }}
              >
                <Stack direction="row" spacing={0.65} sx={{ flex: 1, width: 0, maxWidth: '100%', alignItems: 'center', minWidth: 0, overflow: 'hidden' }}>
                  <AgentExecutionStatusIcon status={node.status || (node.skipped ? 'skipped' : 'completed')} size={16} />
                  <Typography variant="body2" noWrap sx={{ fontWeight: 700, minWidth: 105, maxWidth: 190 }}>{node.label}</Typography>
                  {nodeVisits.length > 1 && (
                    <Stack direction="row" spacing={0.1} alignItems="center" onClick={(event) => event.stopPropagation()} onKeyDown={(event) => event.stopPropagation()}>
                      <Tooltip title={previousVisit ? `Previous ${node.label} invocation` : 'First invocation'} arrow>
                        <span>
                          <IconButton
                            size="small"
                            aria-label={`Previous invocation of ${node.label}`}
                            disabled={!previousVisit}
                            onClick={() => previousVisit && selectVisit(previousVisit, true)}
                            sx={{ width: 23, height: 23, border: 1, borderColor: 'divider' }}
                          >
                            <ChevronLeftIcon sx={{ fontSize: 17 }} />
                          </IconButton>
                        </span>
                      </Tooltip>
                      <Typography variant="caption" sx={{ minWidth: 30, textAlign: 'center', fontWeight: 700 }}>{visitPosition + 1}/{nodeVisits.length}</Typography>
                      <Tooltip title={nextVisit ? `Next ${node.label} invocation` : 'Latest invocation'} arrow>
                        <span>
                          <IconButton
                            size="small"
                            aria-label={`Next invocation of ${node.label}`}
                            disabled={!nextVisit}
                            onClick={() => nextVisit && selectVisit(nextVisit, true)}
                            sx={{ width: 23, height: 23, border: 1, borderColor: 'divider' }}
                          >
                            <ChevronRightIcon sx={{ fontSize: 17 }} />
                          </IconButton>
                        </span>
                      </Tooltip>
                    </Stack>
                  )}
                  <Tooltip title={<Box sx={{ maxWidth: 520, overflowWrap: 'anywhere' }}>{summary}</Box>} placement="top" arrow>
                    <Typography variant="caption" color="text.secondary" noWrap sx={{ flex: 1, minWidth: 0, maxWidth: '100%' }}>{summary}</Typography>
                  </Tooltip>
                  {activeModel && <Chip size="small" color="primary" variant="outlined" label={`Calling ${activeModel.model_name || 'model'}…`} sx={{ height: 21, flexShrink: 0 }} />}
                </Stack>
              </AccordionSummary>
              {expanded === key && <AccordionDetails sx={{ width: '100%', minWidth: 0, maxWidth: '100%', overflowX: 'hidden', px: 1, pt: 0.25, pb: 1 }}>
                <Stack spacing={0.6} sx={{ mb: 0.75 }}>
                  <Stack direction={{ xs: 'column', sm: 'row' }} spacing={0.6} alignItems={{ sm: 'center' }} justifyContent="space-between">
                    <Stack direction="row" spacing={0.5} sx={{ flexWrap: 'wrap', rowGap: 0.5 }}>
                      <Chip size="small" variant="outlined" label={node.type || node.id} sx={{ height: 21 }} />
                      {formattedDuration && <Chip size="small" variant="outlined" label={formattedDuration} sx={{ height: 21 }} />}
                      {route && <Chip size="small" color="primary" variant="outlined" label={route} aria-label={`Route: ${route}`} sx={{ height: 21 }} />}
                      {visitTools.length > 0 && <Chip size="small" variant="outlined" icon={<BuildOutlinedIcon />} label={visitTools.length} sx={{ height: 21 }} />}
                      {visitModels.length > 0 && <Chip size="small" variant="outlined" icon={<MemoryIcon />} label={`${visitModels.length} model${visitModels.length === 1 ? '' : 's'}`} sx={{ height: 21 }} />}
                      {node.warningCodes.length > 0 && <Chip size="small" color="warning" variant="outlined" icon={<WarningAmberIcon />} label={node.warningCodes.length} sx={{ height: 21 }} />}
                      {hasNodeError && <Chip size="small" color="error" variant="outlined" icon={<ErrorOutlineIcon />} label="1" sx={{ height: 21 }} />}
                    </Stack>
                  </Stack>
                  {routeReason && (
                    <Typography component="div" variant="body2" color="text.secondary" sx={{ display: 'block', width: '100%', minWidth: 0, maxWidth: '100%', whiteSpace: 'normal', overflow: 'hidden', overflowWrap: 'anywhere', wordBreak: 'break-word' }}>
                      <Box component="span" sx={{ fontWeight: 700, color: 'text.primary' }}>Route reason: </Box>{routeReason}
                    </Typography>
                  )}
                </Stack>
                {loadingKey === key && <Stack direction="row" spacing={1}><CircularProgress size={16} /><Typography variant="caption">Loading full details…</Typography></Stack>}
                {detailErrors[key] && <Alert severity="info">Older trace—full invocation details are unavailable.</Alert>}
                {detail && <AgentNodeExecutionDetails detail={detail} />}
                {visitModels.length > 0 && (
                  <Stack spacing={0.35} sx={{ mt: 0.7 }}>
                    <Typography variant="caption" sx={{ fontWeight: 700 }}>Model activity</Typography>
                    {visitModels.map((model) => (
                      <Typography key={model.invocation_id || model.event_id} variant="caption" color={model.status === 'failed' ? 'error' : 'text.secondary'}>
                        {model.status === 'started' || model.status === 'active' ? 'Calling' : model.status === 'failed' ? 'Failed' : 'Completed'} {model.model_name || 'model'}
                        {model.duration_ms != null ? ` · ${formatDurationMs(model.duration_ms)}` : ''}
                        {model.retry_count ? ` · ${model.retry_count} retries` : ''}
                        {model.error?.message ? ` · ${model.error.message}` : ''}
                      </Typography>
                    ))}
                  </Stack>
                )}
                {visitTools.length > 0 && (
                  <Stack spacing={0.35} sx={{ mt: 0.7 }}>
                    <Typography variant="caption" sx={{ fontWeight: 700 }}>Tool activity</Typography>
                    {visitTools.map((tool) => (
                      <Typography key={tool.id || `${tool.name}:${tool.callerVisitIndex}`} variant="caption" color={tool.status === 'failed' || tool.ok === false ? 'error' : 'text.secondary'}>
                        {tool.status === 'started' || tool.status === 'progress' ? 'Calling' : tool.status === 'failed' || tool.ok === false ? 'Failed' : 'Completed'} {tool.displayName || tool.name}
                        {tool.durationMs != null ? ` · ${formatDurationMs(tool.durationMs)}` : ''}
                      </Typography>
                    ))}
                  </Stack>
                )}
              </AccordionDetails>}
            </Accordion>
          );
        })}
        {!chatMode && finalOutput?.answer && (
          <Box
            component="details"
            open={finalAnswerOpen}
            onToggle={(event) => setFinalAnswerOpen(event.currentTarget.open)}
            sx={{ mt: 1, border: 1, borderColor: 'divider', borderRadius: 1, bgcolor: 'background.paper' }}
          >
            <Box component="summary" sx={{ cursor: 'pointer', px: 1, py: 0.65, fontSize: '0.82rem', fontWeight: 700 }}>
              Final answer
            </Box>
            {finalAnswerOpen && (
              <Box sx={{ px: 1, pb: 1, minWidth: 0 }}>
                <Stack direction="row" justifyContent="flex-end">
                  <Tooltip title="Copy answer"><IconButton size="small" onClick={() => void copyAnswer()}><ContentCopyIcon fontSize="small" /></IconButton></Tooltip>
                </Stack>
                <Box sx={{ minWidth: 0, maxWidth: '100%', overflowWrap: 'anywhere', wordBreak: 'break-word', '& pre': { maxWidth: '100%', overflowX: 'auto' }, '& table': { display: 'block', maxWidth: '100%', overflowX: 'auto' }, '& a': { overflowWrap: 'anywhere' } }}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{finalOutput.answer}</ReactMarkdown>
                </Box>
              </Box>
            )}
          </Box>
        )}
            </>
          )}
        </Box>
      </Paper>
      <GenericTraceTimeline events={traceView.events} focusedEventId={focusedEventId} />
      <TraceVisualizationSlot
        traceView={traceView}
        resolvedSpec={resolvedSpec}
        framework={framework}
        workflowId={workflowId}
        focusedTraceRefs={focusedTraceRefs}
        selectedVisitRef={selectedTopologyNodeId ? { nodeId: selectedTopologyNodeId, visitIndex: selectedVisit?.visitIndex || 1 } : null}
        onGraphSelection={handleGraphSelection}
        live={running}
      />
    </Stack>
  );
}

export default React.memo(AgentExecutionView);
