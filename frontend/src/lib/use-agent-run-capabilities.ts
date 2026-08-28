import { useCallback, useEffect, useRef, useState } from 'react';
import { getAgentRunCapabilities, type AgentRuntimeCapabilityResponse } from './api';
import { runtimeCapabilityResponseMatchesRun } from './runtime-capabilities';

const RECOVERY_DELAYS_MS = [1000, 2000, 4000, 8000, 16000, 30000] as const;

export function useAgentRunCapabilities(
  runId: string | null | undefined,
  threadId: string | null | undefined,
  refreshKey: string | number = '',
) {
  const [capabilities, setCapabilities] = useState<AgentRuntimeCapabilityResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const requestId = useRef(0);
  const refresh = useCallback(async () => {
    if (!runId || !threadId) return;
    const currentRequest = ++requestId.current;
    try {
      const result = await getAgentRunCapabilities(runId, threadId);
      if (currentRequest !== requestId.current) return;
      if (!runtimeCapabilityResponseMatchesRun(result, runId)) {
        setCapabilities(null);
        setError('Run capabilities did not match the selected run.');
        return;
      }
      setCapabilities(result);
      setError(result.runtime_available ? null : 'The runtime deployment is unavailable. Run controls will remain disabled until it recovers.');
    } catch (value) {
      if (currentRequest !== requestId.current) return;
      setCapabilities(null);
      setError(value instanceof Error ? value.message : String(value));
    }
  }, [runId, threadId]);

  useEffect(() => {
    requestId.current += 1;
    setCapabilities(null);
    setError(null);
    if (!runId || !threadId) return undefined;
    let active = true;
    let timer: number | undefined;
    let recoveryAttempt = 0;

    const load = async () => {
      if (!active) return;
      const currentRequest = ++requestId.current;
      try {
        const result = await getAgentRunCapabilities(runId, threadId);
        if (!active || currentRequest !== requestId.current) return;
        if (!runtimeCapabilityResponseMatchesRun(result, runId)) {
          setCapabilities(null);
          setError('Run capabilities did not match the selected run.');
          return;
        }
        setCapabilities(result);
        setError(result.runtime_available ? null : 'The runtime deployment is unavailable. Run controls will remain disabled until it recovers.');
        if (!result.runtime_available) {
          timer = window.setTimeout(load, RECOVERY_DELAYS_MS[Math.min(recoveryAttempt++, RECOVERY_DELAYS_MS.length - 1)]);
        }
      } catch (value) {
        if (!active || currentRequest !== requestId.current) return;
        setCapabilities(null);
        setError(value instanceof Error ? value.message : String(value));
        timer = window.setTimeout(load, RECOVERY_DELAYS_MS[Math.min(recoveryAttempt++, RECOVERY_DELAYS_MS.length - 1)]);
      }
    };
    void load();
    return () => {
      active = false;
      requestId.current += 1;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [runId, threadId, refreshKey]);

  return { capabilities, error, refresh };
}
