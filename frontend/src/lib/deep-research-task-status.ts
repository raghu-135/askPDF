import type { AgentTaskStatus } from './api';

const TASK_STATUS_LABELS: Record<string, string> = {
  created: 'Created',
  queued: 'Queued',
  running: 'Running',
  pausing: 'Pausing',
  paused: 'Paused',
  awaiting_approval: 'Awaiting your decision',
  cancelling: 'Cancelling',
  recovery_required: 'Recovery required',
  cancelled: 'Cancelled',
  completed: 'Completed',
  failed: 'Failed',
  expired: 'Expired',
};

const RUN_STATUS_LABELS: Record<string, string> = {
  awaiting_human: 'Awaiting your decision',
  running: 'Running',
  queued: 'Queued',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
  expired: 'Expired',
};

export function formatAgentTaskStatus(status: AgentTaskStatus | string): string {
  return TASK_STATUS_LABELS[status] || String(status).replaceAll('_', ' ');
}

export function formatAgentRunStatus(status: string): string {
  return RUN_STATUS_LABELS[status] || status.replaceAll('_', ' ');
}

export function formatTimelineItemStatus(status: string): string {
  return formatAgentRunStatus(status);
}
