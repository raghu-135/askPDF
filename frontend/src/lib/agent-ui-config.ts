import { requiredPositiveMilliseconds } from './api-config';

export const AGENT_TASK_POLL_INTERVAL_MS = requiredPositiveMilliseconds(
  'NEXT_PUBLIC_AGENT_TASK_POLL_INTERVAL_MS',
  process.env.NEXT_PUBLIC_AGENT_TASK_POLL_INTERVAL_MS,
);

export const AGENT_SSE_RECONNECT_INTERVAL_MS = requiredPositiveMilliseconds(
  'NEXT_PUBLIC_AGENT_SSE_RECONNECT_INTERVAL_MS',
  process.env.NEXT_PUBLIC_AGENT_SSE_RECONNECT_INTERVAL_MS,
);
