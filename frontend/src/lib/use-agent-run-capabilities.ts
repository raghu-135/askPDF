import { useCallback, useEffect, useRef, useState } from 'react';
import { getAgentRunCapabilities, type AgentRuntimeCapabilityResponse } from './api';
import { runtimeCapabilityResponseMatchesRun } from './runtime-capabilities';

const RECOVERY_DELAYS_MS = [1000, 2000, 4000, 8000, 16000, 30000] as const;

export function useAgentRunCapabilities(
  runId: string | null | undefined,
  threadId: string | null | undefined,
  refreshKey: string | number = '',
) {
  const identity = JSON.stringify([runId, threadId, refreshKey]);
  const currentIdentity = useRef(identity);
  currentIdentity.current = identity;
  const requestId = useRef(0);
  const [state, setState] = useState<{
    identity: string;
    capabilities: AgentRuntimeCapabilityResponse | null;
    error: string | null;
  } | null>(null);

  const refresh = useCallback(async (): Promise<boolean> => {
    // Normal chat uses an optimistic assistant message id until the runtime
    // returns the durable agent run id. That placeholder is not addressable by
    // the runtime capabilities endpoint.
    if (!runId || runId.startsWith('temp-assistant-') || !threadId) return false;
    const currentRequest = ++requestId.current;
    const isCurrent = () => currentRequest === requestId.current && identity === currentIdentity.current;
    try {
      const result = await getAgentRunCapabilities(runId, threadId);
      if (!isCurrent()) return false;
      if (!runtimeCapabilityResponseMatchesRun(result, runId)) {
        throw new Error('Run capabilities did not match the selected run.');
      }
      setState({
        identity,
        capabilities: result,
        error: result.runtime_available ? null : 'The runtime deployment is unavailable. Run controls will remain disabled until it recovers.',
      });
      return !result.runtime_available;
    } catch (value) {
      if (!isCurrent()) return false;
      setState({ identity, capabilities: null, error: value instanceof Error ? value.message : String(value) });
      return true;
    }
  }, [runId, threadId, identity]);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let recoveryAttempt = 0;
    const load = async () => {
      const retry = await refresh();
      if (active && retry) {
        timer = setTimeout(load, RECOVERY_DELAYS_MS[Math.min(recoveryAttempt++, RECOVERY_DELAYS_MS.length - 1)]);
      }
    };
    void load();
    return () => {
      active = false;
      requestId.current += 1;
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [refresh]);

  const current = state?.identity === identity ? state : null;
  return { capabilities: current?.capabilities ?? null, error: current?.error ?? null, refresh };
}
