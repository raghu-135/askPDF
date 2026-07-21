import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import { Accordion, AccordionDetails, AccordionSummary, Alert, Box, Chip, CircularProgress, IconButton, Paper, Stack, Tooltip, Typography } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import dynamic from 'next/dynamic';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { getAgentRunNodeDetails, type AgentRunNodeDetail } from '../../lib/api';
import type { TraceNodeView, TraceRunView } from '../agent-debug/agent-trace-projection';
import type { AgentGraphSelection, AgentTraceRefs } from './agent-graph-types';
import AgentNodeExecutionDetails from './AgentNodeExecutionDetails';

const AgentDebugCanvas = dynamic(() => import('./AgentDebugCanvas'), { ssr: false });

const visitKey = (node: Pick<TraceNodeView, 'id' | 'visitIndex'>) => `${node.id}:${node.visitIndex || 1}`;

const nodeSummary = (node: TraceNodeView) => {
  const raw = node.raw || {};
  const detail = raw.detail || {};
  const event = detail.event || raw;
  if (node.status === 'error') return `Failed: ${detail.error?.raw_message || detail.error || event.error?.raw_message || event.error || node.error?.raw_message || 'node execution failed'}`;
  if (node.skipped) return `Skipped${event.skip_reason ? `: ${event.skip_reason}` : ''}`;
  if (event.evaluator_route) return `Evaluated evidence and chose ${event.evaluator_route}.`;
  if (event.route) return `Selected the ${event.route} route${event.route_reason ? `: ${event.route_reason}` : '.'}`;
  if (Array.isArray(event.execution_plan)) return `Planned ${event.execution_plan.length} step${event.execution_plan.length === 1 ? '' : 's'}: ${event.execution_plan.join(' → ')}.`;
  if (event.document_source_count || event.web_source_count) return `Retrieved ${Number(event.document_source_count || 0) + Number(event.web_source_count || 0)} source${Number(event.document_source_count || 0) + Number(event.web_source_count || 0) === 1 ? '' : 's'}.`;
  if (event.answer_chars) return `Generated an answer (${event.answer_chars} characters).`;
  return node.status === 'active' ? 'Running…' : 'Completed this step.';
};

export default function AgentExecutionView({
  runId,
  threadId,
  resolvedSpec,
  workflowId,
  traceView,
  status,
  running = false,
  focusedTraceRefs,
}: {
  runId?: string | null;
  threadId?: string | null;
  resolvedSpec?: Record<string, any>;
  workflowId?: string;
  traceView: TraceRunView;
  status?: string;
  running?: boolean;
  focusedTraceRefs?: AgentTraceRefs | null;
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
  const [selectedNodeId, setSelectedNodeId] = useState<string | undefined>();
  const [loadingKey, setLoadingKey] = useState<string | null>(null);
  const [detailErrors, setDetailErrors] = useState<Record<string, string>>({});
  const inFlightDetailKeys = useRef(new Set<string>());
  const detailContextKey = `${runId || 'live'}:${threadId || ''}`;
  const detailContextRef = useRef(detailContextKey);

  useEffect(() => {
    if (detailContextRef.current !== detailContextKey) {
      detailContextRef.current = detailContextKey;
      inFlightDetailKeys.current.clear();
      setDetails(initialDetails);
      setExpanded(false);
      setSelectedNodeId(undefined);
      setLoadingKey(null);
      setDetailErrors({});
      return;
    }
    setDetails((current) => ({ ...current, ...initialDetails }));
  }, [detailContextKey, initialDetails]);

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
    setExpanded(open ? key : false);
    if (open) {
      setSelectedNodeId(node.id);
      void loadDetail(node);
    }
  }, [loadDetail]);

  const handleGraphSelection = useCallback((selection: AgentGraphSelection) => {
    if (!selection || selection.kind !== 'node') return;
    const node = [...traceView.nodes].reverse().find((row) => row.id === selection.node.id);
    if (node) selectVisit(node, true);
  }, [selectVisit, traceView.nodes]);

  const finalOutput = traceView.finalOutput;
  const copyAnswer = async () => {
    if (finalOutput?.answer && navigator.clipboard) await navigator.clipboard.writeText(finalOutput.answer);
  };

  return (
    <Stack spacing={1.5} sx={{ minWidth: 0 }}>
      <Paper variant="outlined" sx={{ p: 1.5 }}>
        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1, flexWrap: 'wrap', rowGap: 0.5 }}>
          <Typography variant="h6">Execution progress</Typography>
          <Chip size="small" label={status || (running ? 'running' : 'completed')} color={running ? 'primary' : 'default'} />
          {traceView.route && <Chip size="small" variant="outlined" label={`Route: ${traceView.route}`} />}
          <Chip size="small" variant="outlined" label={`${traceView.nodes.length} visits`} />
        </Stack>
        {traceView.nodes.length === 0 ? (
          <Typography variant="body2" color="text.secondary">Start a run to see each node invocation.</Typography>
        ) : traceView.nodes.map((node, index) => {
          const key = visitKey(node);
          const detail = details[key];
          return (
            <Accordion key={`${key}:${index}`} expanded={expanded === key} onChange={(_, open) => selectVisit(node, open)} disableGutters>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} sx={{ width: '100%', alignItems: { sm: 'center' }, minWidth: 0 }}>
                  <Typography variant="body2" sx={{ fontWeight: 700, minWidth: 150 }}>{node.label} · Visit {node.visitIndex || 1}</Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ flex: 1 }}>{nodeSummary(node)}</Typography>
                  <Chip size="small" variant="outlined" color={node.status === 'error' ? 'error' : node.status === 'active' ? 'primary' : 'default'} label={node.status || 'completed'} />
                </Stack>
              </AccordionSummary>
              <AccordionDetails>
                {loadingKey === key && <Stack direction="row" spacing={1}><CircularProgress size={16} /><Typography variant="caption">Loading full details…</Typography></Stack>}
                {detailErrors[key] && <Alert severity="info">Older trace—full details unavailable. Existing previews remain available in the graph inspector.</Alert>}
                {detail && <AgentNodeExecutionDetails detail={detail} />}
              </AccordionDetails>
            </Accordion>
          );
        })}
        {finalOutput?.answer && (
          <Box sx={{ mt: 2, p: 1.5, border: 1, borderColor: 'divider', borderRadius: 1, bgcolor: 'background.paper' }}>
            <Stack direction="row" alignItems="center" justifyContent="space-between">
              <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>Final answer</Typography>
              <Tooltip title="Copy answer"><IconButton size="small" onClick={() => void copyAnswer()}><ContentCopyIcon fontSize="small" /></IconButton></Tooltip>
            </Stack>
            <Box sx={{ mt: 1, '& pre': { overflowX: 'auto' }, '& table': { display: 'block', overflowX: 'auto' } }}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{finalOutput.answer}</ReactMarkdown>
            </Box>
          </Box>
        )}
      </Paper>
      <Box sx={{ minHeight: 520 }}>
        <AgentDebugCanvas
          resolvedSpec={resolvedSpec}
          workflowId={workflowId}
          traceView={traceView}
          focusedTraceRefs={focusedTraceRefs}
          selectedNodeId={selectedNodeId}
          selectedNodeDetail={expanded ? details[expanded] : undefined}
          onSelectionChange={handleGraphSelection}
        />
      </Box>
    </Stack>
  );
}
