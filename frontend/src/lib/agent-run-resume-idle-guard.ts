/** Abort a resume SSE when the server only heartbeats for too long. */

export const AGENT_RUN_RESUME_IDLE_TIMEOUT_MS = 120_000;

export type AgentRunResumeIdleGuard = {
  touch: (eventName: string) => void;
  dispose: () => void;
};

export function startAgentRunResumeIdleGuard(
  signal: AbortSignal,
  onIdle: () => void,
  timeoutMs: number = AGENT_RUN_RESUME_IDLE_TIMEOUT_MS,
): AgentRunResumeIdleGuard {
  let lastProgressAt = Date.now();
  const timer = window.setInterval(() => {
    if (signal.aborted) return;
    if (Date.now() - lastProgressAt >= timeoutMs) onIdle();
  }, 2_000);
  return {
    touch: (eventName: string) => {
      if (eventName !== 'heartbeat') lastProgressAt = Date.now();
    },
    dispose: () => window.clearInterval(timer),
  };
}