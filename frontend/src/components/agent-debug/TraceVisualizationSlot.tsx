import React, { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { Box, Chip, Paper, Stack, Typography } from '@mui/material';
import type { AgentTraceVisualization } from '../../lib/api';
import type { TraceRunView } from './agent-trace-projection';
import type { AgentGraphSelection, AgentNodeVisitRef, AgentTraceRefs } from '../agent-graph/agent-graph-types';
import { getAgentGraphSpec } from '../agent-graph/agent-graph-mapper';
import ParallelTraceLanes from './ParallelTraceLanes';

const AgentDebugCanvas = dynamic(() => import('../agent-graph/AgentDebugCanvas'), { ssr: false });

export interface TraceVisualizationProps {
  descriptor: AgentTraceVisualization;
  traceView: TraceRunView;
  resolvedSpec?: Record<string, any>;
  framework?: string;
  workflowId?: string;
  focusedTraceRefs?: AgentTraceRefs | null;
  selectedVisitRef?: AgentNodeVisitRef | null;
  onGraphSelection?: (selection: AgentGraphSelection) => void;
  onEventFocus?: (eventId: string) => void;
  onOperationFocus?: (operationId: string) => void;
  live?: boolean;
}

interface TraceVisualizationSlotProps extends Omit<TraceVisualizationProps, 'descriptor'> {
  visualizationIds?: readonly AgentTraceVisualization['id'][];
  excludeVisualizationIds?: readonly AgentTraceVisualization['id'][];
}

export interface TraceVisualizationProvider {
  id: AgentTraceVisualization['id'];
  render: (props: TraceVisualizationProps) => React.ReactNode;
}

const langGraphProvider: TraceVisualizationProvider = {
  id: 'langgraph.graph',
  render: ({ traceView, resolvedSpec, workflowId, focusedTraceRefs, selectedVisitRef, onGraphSelection }) => (
    <Box sx={{ minHeight: 400, mt: 0.4, mx: -1 }}>
      <AgentDebugCanvas
        resolvedSpec={resolvedSpec}
        workflowId={workflowId}
        traceView={traceView}
        focusedTraceRefs={focusedTraceRefs}
        selectedVisitRef={selectedVisitRef}
        onSelectionChange={onGraphSelection}
      />
    </Box>
  ),
};

const HermesSessionView = ({ descriptor }: { descriptor: AgentTraceVisualization }) => {
  if (descriptor.id !== 'hermes.session') return null;
  const sections = [
    ['Reasoning', descriptor.reasoning || []],
    ['Approvals', descriptor.approvals || []],
    ['Tools', descriptor.tools || []],
    ['Subagents', descriptor.subagents || []],
    ['Failures', descriptor.failures || []],
  ] as const;
  return (
    <Stack spacing={0.75} sx={{ py: 0.5 }}>
      <Stack direction="row" spacing={0.5} flexWrap="wrap">
        {descriptor.session_id && <Chip size="small" label={`Session ${descriptor.session_id}`} />}
        {descriptor.upstream_run_id && <Chip size="small" variant="outlined" label={`Run ${descriptor.upstream_run_id}`} />}
      </Stack>
      {sections.map(([label, events]) => events.length > 0 && (
        <Box key={label}>
          <Typography variant="caption" sx={{ fontWeight: 700 }}>{label} ({events.length})</Typography>
          {events.map((event) => {
            const payload = event.payload || {};
            const identity = payload.subagent_id || payload.tool_name || payload.approval_id || '';
            const parent = payload.parent_subagent_id;
            return (
              <Typography key={event.event_id} component="div" variant="caption" color="text.secondary" sx={{ pl: parent ? 1.5 : 0 }}>
                #{event.sequence} {parent ? `${parent} → ` : ''}{identity ? `${identity} · ` : ''}{event.kind}
              </Typography>
            );
          })}
        </Box>
      ))}
    </Stack>
  );
};

const hermesProvider: TraceVisualizationProvider = {
  id: 'hermes.session',
  render: ({ descriptor }) => <HermesSessionView descriptor={descriptor} />,
};

const parallelProvider: TraceVisualizationProvider = {
  id: 'generic.parallel',
  render: ({ traceView, onEventFocus, onOperationFocus }) => (
    <ParallelTraceLanes
      groups={traceView.parallelGroups}
      onEventFocus={onEventFocus}
      onOperationFocus={onOperationFocus}
      operationLabels={Object.fromEntries(traceView.operations.map((operation) => [operation.id, operation.label]))}
    />
  ),
};

export const TRACE_VISUALIZATION_PROVIDERS: ReadonlyMap<string, TraceVisualizationProvider> = new Map([
  [parallelProvider.id, parallelProvider],
  [langGraphProvider.id, langGraphProvider],
  [hermesProvider.id, hermesProvider],
]);

export default function TraceVisualizationSlot(props: TraceVisualizationSlotProps) {
  const [visualizationTrace, setVisualizationTrace] = useState(props.traceView);
  useEffect(() => {
    if (!props.live) {
      setVisualizationTrace(props.traceView);
      return undefined;
    }
    const timer = window.setTimeout(() => setVisualizationTrace(props.traceView), 200);
    return () => window.clearTimeout(timer);
  }, [props.live, props.traceView]);
  const visualizations = { ...visualizationTrace.visualizations };
  const normalizedFramework = String(props.framework || '').trim().toLowerCase().replace(/[-\s]+/g, '_');
  // Keep the provider descriptor after termination as well. The retained
  // payload is authoritative for final operation state, while the resolved
  // definition remains the stable source of graph topology.
  const graphSpec = getAgentGraphSpec(props.resolvedSpec, props.workflowId);
  const canProjectLiveLangGraph = normalizedFramework === 'langgraph'
    || normalizedFramework === 'deep_agents';
  if (!visualizations['langgraph.graph'] && canProjectLiveLangGraph && graphSpec?.nodes?.length) {
    // The topology is already available on the run definition. Publish only a
    // lightweight provider descriptor here; live operation/tool state remains
    // in the canonical trace view and is overlaid by the graph provider.
    visualizations['langgraph.graph'] = { id: 'langgraph.graph' };
  }
  const matches = Object.entries(visualizations)
    .filter(([id]) => TRACE_VISUALIZATION_PROVIDERS.has(id))
    .filter(([id]) => !props.visualizationIds || props.visualizationIds.includes(id as AgentTraceVisualization['id']))
    .filter(([id]) => !props.excludeVisualizationIds?.includes(id as AgentTraceVisualization['id']));
  if (matches.length === 0) return null;
  return (
    <>
      {matches.map(([id, descriptor]) => {
        const provider = TRACE_VISUALIZATION_PROVIDERS.get(id)!;
        return (
          <Paper key={id} elevation={0} square sx={{ px: 1, py: 0.4, borderTop: 1, borderColor: 'divider', bgcolor: 'background.default' }}>
            <Box component="details" open={id === 'generic.parallel'}>
              <Box component="summary" sx={{ cursor: 'pointer', py: 0.35, fontSize: '0.78rem', fontWeight: 700 }}>
                {id === 'langgraph.graph' ? 'Execution graph' : id === 'generic.parallel' ? 'Parallel execution' : 'Hermes session'}
              </Box>
                {provider.render({ ...props, traceView: visualizationTrace, descriptor, live: props.live })}
            </Box>
          </Paper>
        );
      })}
    </>
  );
}
