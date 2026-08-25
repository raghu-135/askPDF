import React from 'react';
import dynamic from 'next/dynamic';
import { Box, Chip, Paper, Stack, Typography } from '@mui/material';
import type { AgentTraceVisualization } from '../../lib/api';
import type { TraceRunView } from './agent-trace-projection';
import type { AgentGraphSelection, AgentNodeVisitRef, AgentTraceRefs } from '../agent-graph/agent-graph-types';

const AgentDebugCanvas = dynamic(() => import('../agent-graph/AgentDebugCanvas'), { ssr: false });

export interface TraceVisualizationProps {
  descriptor: AgentTraceVisualization;
  traceView: TraceRunView;
  resolvedSpec?: Record<string, any>;
  workflowId?: string;
  focusedTraceRefs?: AgentTraceRefs | null;
  selectedVisitRef?: AgentNodeVisitRef | null;
  onGraphSelection?: (selection: AgentGraphSelection) => void;
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

export const TRACE_VISUALIZATION_PROVIDERS: ReadonlyMap<string, TraceVisualizationProvider> = new Map([
  [langGraphProvider.id, langGraphProvider],
  [hermesProvider.id, hermesProvider],
]);

export default function TraceVisualizationSlot(props: Omit<TraceVisualizationProps, 'descriptor'>) {
  const matches = Object.entries(props.traceView.visualizations)
    .filter(([id]) => TRACE_VISUALIZATION_PROVIDERS.has(id));
  if (matches.length === 0) return null;
  return (
    <>
      {matches.map(([id, descriptor]) => {
        const provider = TRACE_VISUALIZATION_PROVIDERS.get(id)!;
        return (
          <Paper key={id} elevation={0} square sx={{ px: 1, py: 0.4, borderTop: 1, borderColor: 'divider', bgcolor: 'background.default' }}>
            <Box component="details" open={id === 'langgraph.graph'}>
              <Box component="summary" sx={{ cursor: 'pointer', py: 0.35, fontSize: '0.78rem', fontWeight: 700 }}>
                {id === 'langgraph.graph' ? 'Execution graph' : 'Hermes session'}
              </Box>
              {provider.render({ ...props, descriptor })}
            </Box>
          </Paper>
        );
      })}
    </>
  );
}
