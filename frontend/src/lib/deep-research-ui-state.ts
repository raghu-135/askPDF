import type { AgentRunDetails, AgentTaskRun, AgentTaskSummary } from './api';

const TERMINAL_TASK_STATUSES = new Set(['completed', 'failed', 'expired', 'cancelled']);

export function mergeActiveAgentTaskRun(task: AgentTaskSummary, runs: AgentTaskRun[]): AgentTaskRun[] {
  const activeRun = task.active_run;
  if (!activeRun) return runs;
  return runs.map((run) => run.id === activeRun.id ? {
    ...run,
    status: activeRun.status || run.status,
    pending_interrupt: activeRun.pending_interrupt ?? run.pending_interrupt,
  } : run);
}

export function shouldPollAgentTask(task: AgentTaskSummary | null): boolean {
  return Boolean(task && !TERMINAL_TASK_STATUSES.has(task.status));
}

export function isTaskOwnedAgentRun(run: AgentRunDetails | undefined): boolean {
  return Boolean(run?.task_id);
}
