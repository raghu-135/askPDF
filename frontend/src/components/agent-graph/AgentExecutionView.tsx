import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import AccountTreeOutlinedIcon from '@mui/icons-material/AccountTreeOutlined';
import BuildOutlinedIcon from '@mui/icons-material/BuildOutlined';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import TimerOutlinedIcon from '@mui/icons-material/TimerOutlined';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import { Accordion, AccordionDetails, AccordionSummary, Alert, Box, Chip, CircularProgress, IconButton, Paper, Stack, Tooltip, Typography } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import dynamic from 'next/dynamic';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { getAgentRunNodeDetails, type AgentRunNodeDetail } from '../../lib/api';
import type { TraceNodeView, TraceRunView } from '../agent-debug/agent-trace-projection';
import type { AgentGraphSelection, AgentNodeVisitRef, AgentTraceRefs } from './agent-graph-types';
import AgentNodeExecutionDetails from './AgentNodeExecutionDetails';
import AgentExecutionStatusIcon from './AgentExecutionStatusIcon';
import { formatDurationMs } from '../../lib/formatDuration';
import { TraceLlmUsageTooltip, TraceNodesTooltip, TraceToolsTooltip } from '../agent-debug/AgentRunTraceTooltips';
import { compactExecutionText } from './agent-execution-display';
import {
  agentNodeVisitKey,
  getChronologicalNodeVisits,
  getNextNodeVisit,
  getNodeVisitRoute,
  getPreviousNodeVisit,
  toAgentNodeVisitRef,
} from './agent-node-visits';

const AgentDebugCanvas = dynamic(() => import('./AgentDebugCanvas'), { ssr: false });

const visitKey = (node: Pick<TraceNodeView, 'id' | 'visitIndex'>) => agentNodeVisitKey(node);

const nodeSummary = (node: TraceNodeView) => {
  const raw = node.raw || {};
  const detail = raw.detail || {};
  const event = detail.event || raw;
  let summary: unknown;
  if (node.status === 'error') summary = `Failed: ${detail.error?.raw_message || event.error?.raw_message || node.error?.raw_message || 'node execution failed'}`;
  else if (node.skipped) summary = `Skipped${event.skip_reason ? `: ${event.skip_reason}` : ''}`;
  else if (event.evaluator_route) summary = `Evaluated evidence and chose ${event.evaluator_route}.`;
  else if (event.route) summary = `Selected the ${event.route} route${event.route_reason ? `: ${event.route_reason}` : '.'}`;
  else if (Array.isArray(event.execution_plan)) summary = `Planned ${event.execution_plan.length} step${event.execution_plan.length === 1 ? '' : 's'}: ${event.execution_plan.join(' → ')}.`;
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
  workflowId,
  traceView,
  status,
  running = false,
  focusedTraceRefs,
  defaultGraphOpen = false,
}: {
  runId?: string | null;
  threadId?: string | null;
  resolvedSpec?: Record<string, any>;
  workflowId?: string;
  traceView: TraceRunView;
  status?: string;
  running?: boolean;
  focusedTraceRefs?: AgentTraceRefs | null;
  defaultGraphOpen?: boolean;
}) {
  const initialDetails = useMemo(() => {
    const result: Record<string, AgentRunNodeDetail> = {};
    traceView.nodes.forEach((node) => {
      if (node.raw?.detail) result[visitKey(node)] = node.raw.detail as AgentRunNodeDetail;
    });
    return result;
  }, [traceView.nodes]);
  const [details, setDetails] = useState<Record<string, AgentRunNodeDetail>>(initialDetails);
  const [expanded, setExpanded] = useState<string | false>(false);
  const [selectedVisit, setSelectedVisit] = useState<AgentNodeVisitRef | null>(null);
  const [loadingKey, setLoadingKey] = useState<string | null>(null);
  const [detailErrors, setDetailErrors] = useState<Record<string, string>>({});
  const [revealRequest, setRevealRequest] = useState<{ key: string; token: number } | null>(null);
  const [graphOpen, setGraphOpen] = useState(defaultGraphOpen);
  const inFlightDetailKeys = useRef(new Set<string>());
  const timelineRows = useRef(new Map<string, HTMLElement>());
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
      setRevealRequest(null);
      setLoadingKey(null);
      setDetailErrors({});
      setGraphOpen(defaultGraphOpen);
      return;
    }
    setDetails((current) => {
      const changed = Object.entries(initialDetails).some(([key, detail]) => current[key] !== detail);
      return changed ? { ...current, ...initialDetails } : current;
    });
  }, [defaultGraphOpen, detailContextKey, initialDetails]);

  const loadDetail = useCallback(async (node: TraceNodeView) => {
    const key = visitKey(node);
    const requestKey = `${detailContextKey}:${key}`;
    if (details[key] || inFlightDetailKeys.current.has(requestKey) || !runId || !threadId || node.status === 'active') return;
    inFlightDetailKeys.current.add(requestKey);
    setLoadingKey(key);
    setDetailErrors((current) => ({ ...current, [key]: '' }));
    try {
      const detail = await getAgentRunNodeDetails(runId, threadId, node.id, node.visitIndex || 1);
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
  }, [detailContextKey, details, runId, threadId]);

  const selectVisit = useCallback((node: TraceNodeView, open: boolean) => {
    const key = visitKey(node);
    selectionContextRef.current = detailContextKey;
    setExpanded(open ? key : false);
    setSelectedVisit(toAgentNodeVisitRef(node));
    if (open) {
      void loadDetail(node);
    }
  }, [detailContextKey, loadDetail]);

  const revealVisit = useCallback((visit: AgentNodeVisitRef) => {
    const node = traceView.nodes.find((row) => visitKey(row) === agentNodeVisitKey(visit));
    if (!node) return;
    selectVisit(node, true);
    setRevealRequest((current) => ({ key: agentNodeVisitKey(visit), token: (current?.token || 0) + 1 }));
  }, [selectVisit, traceView.nodes]);

  useEffect(() => {
    if (!revealRequest || expanded !== revealRequest.key) return;
    // Wait until the accordion and its details have committed before measuring the row.
    const firstFrame = window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        timelineRows.current.get(revealRequest.key)?.scrollIntoView({ behavior: 'smooth', block: 'start', inline: 'nearest' });
      });
    });
    return () => window.cancelAnimationFrame(firstFrame);
  }, [expanded, revealRequest]);

  const handleGraphSelection = useCallback((selection: AgentGraphSelection) => {
    if (!selection || selection.kind !== 'node') {
      setSelectedVisit(null);
      return;
    }
    const node = [...traceView.nodes].reverse().find((row) => row.id === selection.node.id);
    if (node) revealVisit(toAgentNodeVisitRef(node));
    else setSelectedVisit(null);
  }, [revealVisit, traceView.nodes]);

  useEffect(() => {
    if (!selectedVisit) return;
    if (selectionContextRef.current !== detailContextKey) return;
    const selectedKey = agentNodeVisitKey(selectedVisit);
    if (traceView.nodes.some((node) => visitKey(node) === selectedKey)) return;
    const fallback = [...traceView.nodes].reverse().find((node) => node.id === selectedVisit.nodeId);
    if (fallback) {
      setSelectedVisit(toAgentNodeVisitRef(fallback));
    } else {
      setSelectedVisit(null);
    }
  }, [detailContextKey, selectedVisit, traceView.nodes]);

  const finalOutput = traceView.finalOutput;
  const runDuration = formatDurationMs(Number(traceView.metrics.duration_ms));
  const tokenCount = formatTokenCount(traceView.metrics.llm_token_count_total);
  const copyAnswer = async () => {
    if (finalOutput?.answer && navigator.clipboard) await navigator.clipboard.writeText(finalOutput.answer);
  };

  return (
    <Stack spacing={1} sx={{ width: '100%', minWidth: 0, maxWidth: '100%', overflowX: 'hidden' }}>
      <Paper variant="outlined" sx={{ width: '100%', minWidth: 0, maxWidth: '100%', overflowX: 'hidden', p: 1 }}>
        <Stack direction="row" spacing={0.6} alignItems="center" sx={{ mb: 0.75, flexWrap: 'wrap', rowGap: 0.45 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 700, mr: 0.25 }}>Execution progress</Typography>
          <AgentExecutionStatusIcon status={status || (running ? 'running' : 'completed')} size={17} />
          {traceView.route && (
            <Tooltip title={compactExecutionText(traceView.routeReason || `Selected route: ${traceView.route}`, 320)} arrow>
              <Chip size="small" variant="outlined" label={traceView.route} sx={{ height: 22 }} />
            </Tooltip>
          )}
          {runDuration && <Chip size="small" variant="outlined" icon={<TimerOutlinedIcon />} label={runDuration} sx={{ height: 22 }} />}
          <Tooltip title={<TraceNodesTooltip nodes={traceView.nodes} usedCount={traceView.usedNodeCount} availableCount={traceView.availableNodeCount} />} arrow>
            <Chip aria-label={`${traceView.usedNodeCount} nodes, ${traceView.nodes.length} visits`} size="small" variant="outlined" icon={<AccountTreeOutlinedIcon />} label={`${traceView.usedNodeCount}n${traceView.nodes.length !== traceView.usedNodeCount ? ` · ${traceView.nodes.length}v` : ''}`} sx={{ height: 22 }} />
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
        {traceView.nodes.length === 0 ? (
          <Typography variant="body2" color="text.secondary">Start a run to see each node invocation.</Typography>
        ) : traceView.nodes.map((node, index) => {
          const key = visitKey(node);
          const detail = details[key];
          const nodeVisits = getChronologicalNodeVisits(traceView.nodes, node.id);
          const visitPosition = nodeVisits.findIndex((visit) => visitKey(visit) === key);
          const visitRef = toAgentNodeVisitRef(node);
          const previousVisit = getPreviousNodeVisit(traceView.nodes, visitRef);
          const nextVisit = getNextNodeVisit(traceView.nodes, visitRef);
          const route = getNodeVisitRoute(node);
          const summary = nodeSummary(node);
          const routeReason = compactExecutionText(node.routeReason, 480);
          const hasNodeError = Boolean(node.error && Object.keys(node.error).length > 0);
          const visitTools = traceView.tools.filter((tool) => (
            tool.callerNode === node.id && Number(tool.callerVisitIndex || 1) === visitRef.visitIndex
          ));
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
                minWidth: 0,
                maxWidth: '100%',
                overflow: 'hidden',
                scrollMarginTop: 16,
                ...(selectedVisit && agentNodeVisitKey(selectedVisit) === key ? { borderColor: 'primary.main' } : {}),
              }}
            >
              <AccordionSummary
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
                </Stack>
              </AccordionSummary>
              <AccordionDetails sx={{ width: '100%', minWidth: 0, maxWidth: '100%', overflowX: 'hidden', px: 1, pt: 0.25, pb: 1 }}>
                <Stack spacing={0.6} sx={{ mb: 0.75 }}>
                  <Stack direction={{ xs: 'column', sm: 'row' }} spacing={0.6} alignItems={{ sm: 'center' }} justifyContent="space-between">
                    <Stack direction="row" spacing={0.5} sx={{ flexWrap: 'wrap', rowGap: 0.5 }}>
                      <Chip size="small" variant="outlined" label={node.type || node.id} sx={{ height: 21 }} />
                      {node.durationMs !== undefined && <Chip size="small" variant="outlined" label={formatDurationMs(node.durationMs)} sx={{ height: 21 }} />}
                      {route && <Tooltip title={routeReason || `Selected route: ${route}`} arrow><Chip size="small" color="primary" variant="outlined" label={route} sx={{ height: 21 }} /></Tooltip>}
                      {visitTools.length > 0 && <Chip size="small" variant="outlined" icon={<BuildOutlinedIcon />} label={visitTools.length} sx={{ height: 21 }} />}
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
              </AccordionDetails>
            </Accordion>
          );
        })}
        {finalOutput?.answer && (
          <Box sx={{ mt: 1, p: 1, border: 1, borderColor: 'divider', borderRadius: 1, bgcolor: 'background.paper' }}>
            <Stack direction="row" alignItems="center" justifyContent="space-between">
              <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>Final answer</Typography>
              <Tooltip title="Copy answer"><IconButton size="small" onClick={() => void copyAnswer()}><ContentCopyIcon fontSize="small" /></IconButton></Tooltip>
            </Stack>
            <Box sx={{ mt: 1, minWidth: 0, maxWidth: '100%', overflowWrap: 'anywhere', wordBreak: 'break-word', '& pre': { maxWidth: '100%', overflowX: 'auto' }, '& table': { display: 'block', maxWidth: '100%', overflowX: 'auto' }, '& a': { overflowWrap: 'anywhere' } }}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{finalOutput.answer}</ReactMarkdown>
            </Box>
          </Box>
        )}
      </Paper>
      <Paper variant="outlined" sx={{ px: 1, py: 0.4 }}>
        <Box component="details" open={graphOpen} onToggle={(event) => setGraphOpen(event.currentTarget.open)}>
          <Box component="summary" sx={{ cursor: 'pointer', py: 0.35, fontSize: '0.78rem', fontWeight: 700 }}>
            Execution graph
          </Box>
          {graphOpen && (
            <Box sx={{ minHeight: 400, mt: 0.4 }}>
              <AgentDebugCanvas
                resolvedSpec={resolvedSpec}
                workflowId={workflowId}
                traceView={traceView}
                focusedTraceRefs={focusedTraceRefs}
                selectedVisitRef={selectedVisit}
                onSelectionChange={handleGraphSelection}
              />
            </Box>
          )}
        </Box>
      </Paper>
    </Stack>
  );
}

export default React.memo(AgentExecutionView);
