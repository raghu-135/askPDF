import type { AgentRunDetails, AgentTaskRun, AgentTaskSummary } from './api';

const TERMINAL_TASK_STATUSES = new Set(['completed', 'failed', 'expired', 'cancelled']);
const TERMINAL_RUN_STATUSES = new Set(['completed', 'failed', 'cancelled']);
const QUIESCENT_TASK_STATUSES = new Set([...TERMINAL_TASK_STATUSES, 'recovery_required']);
const QUIESCENT_RUN_STATUSES = new Set([...TERMINAL_RUN_STATUSES, 'recovery_required']);
const TERMINAL_EVENT_TYPES = new Set(['run.completed', 'run.failed', 'run.cancelled']);

export function mergeActiveAgentTaskRun(task: AgentTaskSummary, runs: AgentTaskRun[]): AgentTaskRun[] {
  const activeRun = task.active_run;
  if (!activeRun) return runs;
  return runs.map((run) => run.id === activeRun.id ? {
    ...run,
    status: activeRun.status || run.status,
    runtime_binding_status: activeRun.runtime_binding_status ?? run.runtime_binding_status,
    pending_interrupt: activeRun.pending_interrupt ?? run.pending_interrupt,
  } : run);
}

export function shouldPollAgentTask(task: AgentTaskSummary | null): boolean {
  return Boolean(task && !QUIESCENT_TASK_STATUSES.has(task.status));
}

export function shouldSubscribeToAgentTaskEvents(
  task: AgentTaskSummary | null,
  run: AgentTaskRun | null,
): boolean {
  return Boolean(
    task
    && run
    && !QUIESCENT_TASK_STATUSES.has(task.status)
    && !QUIESCENT_RUN_STATUSES.has(run.status),
  );
}

export function isRunOwnedBySelectedTask(taskId: string | null, run: AgentTaskRun | null): boolean {
  return Boolean(taskId && run && run.task_id === taskId);
}

export function isTerminalAgentTaskEvent(payload: Record<string, unknown>): boolean {
  return payload.terminal === true || TERMINAL_EVENT_TYPES.has(String(payload.type || ''));
}

export function shouldRefreshAgentTaskTimeline(payload: Record<string, unknown>): boolean {
  const type = String(payload.type || '');
  return isTerminalAgentTaskEvent(payload)
    || /^(runtime\.event|subagent\.|artifact\.|output\.)/.test(type);
}

export function isTaskOwnedAgentRun(run: AgentRunDetails | undefined): boolean {
  return Boolean(run?.task_id);
}
