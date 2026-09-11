import React from 'react';
import { Box, Chip, IconButton, Tab, Tabs, Tooltip, Typography } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import RouteIcon from '@mui/icons-material/Route';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import AgentRunDebugPanel from '../agent-debug/AgentRunDebugPanel';
import type { AgentRunDetails, AgentRunResumeAction, AgentTraceRefs } from '../../lib/api';
import type { TraceRunView } from '../agent-debug/agent-trace-projection';

export type TraceRunTab = {
  id: string;
  threadId?: string;
  messageId?: string;
  label: string;
  status?: string;
  routeReason?: string;
  traceRefs?: AgentTraceRefs | null;
  runDetails?: AgentRunDetails;
  liveTraceView?: TraceRunView;
  loading?: boolean;
  error?: string;
  running?: boolean;
  onRunDetailsChange?: (run: AgentRunDetails) => void;
  onResumeAction?: (action: AgentRunResumeAction, selectedOptionIds?: string[]) => Promise<boolean>;
};

export default function TraceWorkspace({
  tabs,
  activeRunId,
  onActiveRunChange,
  onClose,
  onBackToMessage,
  suspendHeavyContent = false,
}: {
  tabs: TraceRunTab[];
  activeRunId: string | null;
  onActiveRunChange: (runId: string) => void;
  onClose: (runId: string) => void;
  onBackToMessage?: (messageId: string) => void;
  suspendHeavyContent?: boolean;
}) {
  const activeTab = tabs.find((tab) => tab.id === activeRunId) || tabs[0];
  if (!activeTab) {
    return (
      <Box sx={{ height: '100%', display: 'grid', placeItems: 'center', p: 3, color: 'text.secondary' }}>
        <Box sx={{ textAlign: 'center' }}>
          <RouteIcon sx={{ fontSize: 42, opacity: 0.45 }} />
          <Typography variant="h6">No trace open</Typography>
          <Typography variant="body2">Open a trace from an assistant response to inspect its run.</Typography>
        </Box>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', minHeight: 0, display: 'grid', gridTemplateRows: 'auto minmax(0, 1fr)' }}>
      <Tabs value={tabs.findIndex((tab) => tab.id === activeTab.id)} onChange={(_, index) => tabs[index] && onActiveRunChange(tabs[index].id)} variant="scrollable" scrollButtons="auto" aria-label="Open traces" sx={{ minHeight: 38, borderBottom: 1, borderColor: 'divider' }}>
        {tabs.map((tab) => (
          <Tab
            key={tab.id}
            sx={{ minHeight: 38, textTransform: 'none', py: 0 }}
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 0 }}>
                <Typography variant="body2" noWrap sx={{ maxWidth: 170 }}>{tab.label}</Typography>
                {tab.running && <Chip size="small" color="primary" label="Live" sx={{ height: 20 }} />}
                {tab.error && <Chip size="small" color="error" label="Failed" sx={{ height: 20 }} />}
                <Tooltip title="Close trace"><IconButton size="small" onClick={(event) => { event.stopPropagation(); onClose(tab.id); }} sx={{ p: 0.2 }}><CloseIcon sx={{ fontSize: 15 }} /></IconButton></Tooltip>
              </Box>
            }
          />
        ))}
      </Tabs>
      <Box sx={{ minHeight: 0, overflow: 'auto', bgcolor: 'background.default' }}>
        {activeTab.messageId && onBackToMessage && (
          <Box sx={{ px: 1, py: 0.5 }}><Tooltip title="Return to the originating message"><IconButton size="small" onClick={() => onBackToMessage(activeTab.messageId!)}><ArrowBackIcon fontSize="small" /></IconButton></Tooltip></Box>
        )}
        <AgentRunDebugPanel
          runId={activeTab.id}
          threadId={activeTab.threadId}
          routeReason={activeTab.routeReason}
          traceRefs={activeTab.traceRefs}
          runDetails={activeTab.runDetails}
          loading={activeTab.loading}
          error={activeTab.error}
          liveTraceView={activeTab.liveTraceView}
          running={Boolean(activeTab.running)}
          suspendHeavyContent={suspendHeavyContent}
          onRunDetailsChange={activeTab.onRunDetailsChange}
          onResumeAction={activeTab.onResumeAction}
        />
      </Box>
    </Box>
  );
}
