export type AgentExecutionStatusKind =
  | 'completed'
  | 'failed'
  | 'skipped'
  | 'active'
  | 'interrupted'
  | 'cancelled'
  | 'planned'
  | 'inactive';

export interface AgentExecutionStatusPresentation {
  kind: AgentExecutionStatusKind;
  label: string;
  color: 'success' | 'error' | 'warning' | 'primary' | 'disabled';
  icon: 'check' | 'cross' | 'minus' | 'spinner' | 'pause' | 'clock' | 'dot';
}

export const normalizeAgentExecutionStatus = (status: unknown): AgentExecutionStatusKind => {
  const value = String(status || '').trim().toLowerCase();
  if (['completed', 'complete', 'success', 'succeeded', 'clarification'].includes(value)) return 'completed';
  if (['failed', 'failure', 'error', 'rejected'].includes(value)) return 'failed';
  if (value === 'skipped') return 'skipped';
  if (['active', 'running', 'started'].includes(value)) return 'active';
  if (['interrupted', 'awaiting_human', 'paused'].includes(value)) return 'interrupted';
  if (['cancelled', 'canceled', 'expired'].includes(value)) return 'cancelled';
  if (value === 'planned') return 'planned';
  return 'inactive';
};

export const getAgentExecutionStatusPresentation = (status: unknown): AgentExecutionStatusPresentation => {
  const kind = normalizeAgentExecutionStatus(status);
  const presentations: Record<AgentExecutionStatusKind, Omit<AgentExecutionStatusPresentation, 'kind'>> = {
    completed: { label: 'Completed', color: 'success', icon: 'check' },
    failed: { label: 'Failed', color: 'error', icon: 'cross' },
    skipped: { label: 'Skipped', color: 'warning', icon: 'minus' },
    active: { label: 'Running', color: 'primary', icon: 'spinner' },
    interrupted: { label: 'Awaiting human input', color: 'warning', icon: 'pause' },
    cancelled: { label: 'Cancelled', color: 'disabled', icon: 'minus' },
    planned: { label: 'Planned', color: 'primary', icon: 'clock' },
    inactive: { label: 'Not visited', color: 'disabled', icon: 'dot' },
  };
  return { kind, ...presentations[kind] };
};
