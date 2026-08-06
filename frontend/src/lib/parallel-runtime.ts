export const ParallelRuntimeEvent = {
  DispatchPlanned: 'dispatch.planned',
  DispatchStarted: 'dispatch.started',
  WorkerQueued: 'worker.queued',
  WorkerStarted: 'worker.started',
  WorkerProgress: 'worker.progress',
  WorkerRetrying: 'worker.retrying',
  WorkerCompleted: 'worker.completed',
  WorkerSkipped: 'worker.skipped',
  WorkerFailed: 'worker.failed',
  WorkerTimedOut: 'worker.timed_out',
  WorkerCancelled: 'worker.cancelled',
  BarrierReached: 'dispatch.barrier_reached',
  AggregationCompleted: 'aggregation.completed',
  AggregationPartial: 'aggregation.partial',
  DispatchCancelled: 'dispatch.cancelled',
} as const;

export const ParallelWorkerStatus = {
  Queued: 'queued', Active: 'active', Retrying: 'retrying', Completed: 'completed',
  Skipped: 'skipped', Failed: 'failed', TimedOut: 'timed_out', Cancelled: 'cancelled',
} as const;

export const PARALLEL_TERMINAL_WORKER_STATUSES = new Set<string>([
  ParallelWorkerStatus.Completed, ParallelWorkerStatus.Skipped, ParallelWorkerStatus.Failed,
  ParallelWorkerStatus.TimedOut, ParallelWorkerStatus.Cancelled,
]);

export const PARALLEL_WORKER_STATUS_LABELS: Record<string, string> = {
  [ParallelWorkerStatus.Queued]: 'queued',
  [ParallelWorkerStatus.Active]: 'active',
  [ParallelWorkerStatus.Retrying]: 'retrying',
  [ParallelWorkerStatus.Completed]: 'completed',
  [ParallelWorkerStatus.Skipped]: 'skipped',
  [ParallelWorkerStatus.Failed]: 'failed',
  [ParallelWorkerStatus.TimedOut]: 'timed out',
  [ParallelWorkerStatus.Cancelled]: 'cancelled',
};

export const parallelWorkerStatusForEvent = (event: string): string => {
  if (event === ParallelRuntimeEvent.WorkerQueued) return ParallelWorkerStatus.Queued;
  if (event === ParallelRuntimeEvent.WorkerRetrying) return ParallelWorkerStatus.Retrying;
  if (event === ParallelRuntimeEvent.WorkerStarted || event === ParallelRuntimeEvent.WorkerProgress) return ParallelWorkerStatus.Active;
  return event.startsWith('worker.') ? event.slice('worker.'.length) : '';
};
