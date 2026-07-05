import React from 'react';
import dynamic from 'next/dynamic';
import { Box, CircularProgress, Typography } from '@mui/material';
import type { AgentRunDetails } from '../../lib/api';
import type { AgentGraphRuntimeOverlay } from '../agent-graph/agent-graph-types';
import AgentRunHeaderChips from './AgentRunHeaderChips';
import { getRunDebugMetrics, getRunNodeEvents, getRunToolEvents, getRunTrace } from './agent-debug-utils';

const AgentGraphCanvas = dynamic(() => import('../agent-graph/AgentGraphCanvas'), { ssr: false });

export default function AgentRunDebugPanel({
  runId,
  routeReason,
  runDetails,
  loading,
  error,
}: {
  runId: string;
  routeReason?: string;
  runDetails?: AgentRunDetails;
  loading?: boolean;
  error?: string;
}) {
  const debug = runDetails?.debug;
  const trace = runDetails ? getRunTrace(runDetails) : undefined;
  const metrics = runDetails ? getRunDebugMetrics(runDetails) : {};
  const overlay: AgentGraphRuntimeOverlay = {
    route: trace?.attributes?.['askpdf.route'] || debug?.route || metrics.route,
    routeReason: trace?.attributes?.['askpdf.route_reason'] || debug?.route_reason,
    nodeEvents: runDetails ? getRunNodeEvents(runDetails) : [],
    toolEvents: runDetails ? getRunToolEvents(runDetails) : [],
    errors: debug?.error ? [debug.error] : [],
    metrics,
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
      {debug && runDetails && (
        <>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
            <AgentRunHeaderChips runDetails={runDetails} />
          </Box>
          <AgentGraphCanvas
            resolvedSpec={runDetails.resolved_spec_json}
            templateId={runDetails.template_id}
            mode="run-debug"
            overlay={overlay}
          />
        </>
      )}
    </Box>
  );
}
