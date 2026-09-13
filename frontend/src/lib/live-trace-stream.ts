import type { AgentExecutionStreamEnvelope } from './agent-execution-stream';

export const LIVE_TRACE_TERMINAL_EVENTS = new Set([
  'run.completed',
  'run.failed',
  'run.cancelled',
  'interrupt.requested',
]);

export const isLiveTraceTerminalEvent = (eventName: string) => (
  LIVE_TRACE_TERMINAL_EVENTS.has(eventName)
);

export const liveTraceStatusFromEvent = (
  eventName: string,
  terminalError?: string,
  fallbackStatus?: string,
) => {
  if (terminalError || eventName === 'run.failed') return 'failed';
  if (eventName === 'interrupt.requested') return 'review';
  if (eventName === 'run.cancelled') return 'cancelled';
  if (eventName === 'run.completed') return fallbackStatus || 'completed';
  return fallbackStatus || 'running';
};

export type LiveTraceStreamSnapshot = {
  runId: string;
  events: AgentExecutionStreamEnvelope[];
  running: boolean;
  status: string;
};

export class LiveTraceStreamController {
  private runId: string;
  private events: AgentExecutionStreamEnvelope[] = [];

  constructor(initialRunId: string) {
    this.runId = initialRunId;
  }

  append(event: AgentExecutionStreamEnvelope, terminalError?: string, fallbackStatus?: string): LiveTraceStreamSnapshot {
    if (event.data?.run_id) this.runId = String(event.data.run_id);
    if (event.event !== 'heartbeat') this.events.push(event);
    return this.snapshot(event.event, terminalError, fallbackStatus);
  }

  snapshot(eventName = 'run.started', terminalError?: string, fallbackStatus?: string): LiveTraceStreamSnapshot {
    return {
      runId: this.runId,
      events: [...this.events],
      running: !isLiveTraceTerminalEvent(eventName),
      status: liveTraceStatusFromEvent(eventName, terminalError, fallbackStatus),
    };
  }
}
