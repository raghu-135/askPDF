import type { AgentRunDetails } from './api';
import { InterruptStatus } from './enums';

export function isInterruptAlreadyResolvedError(message: unknown): boolean {
  const text = String(message ?? '').toLowerCase();
  return text.includes('interrupt_already_resolved')
    || text.includes('already been resolved');
}

/** After a failed resume stream, avoid resurrecting a stale approval card. */
export function shouldDismissRecoveredHumanReview(run: AgentRunDetails | null | undefined): boolean {
  if (!run) return true;
  const interrupt = run.pending_interrupt;
  if (!interrupt) return true;
  const status = String(interrupt.status ?? InterruptStatus.Pending);
  return status !== InterruptStatus.Pending;
}