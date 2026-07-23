import { useCallback, useEffect, useRef, useState } from 'react';
import type { AgentExecutionStreamEnvelope } from '../../lib/agent-execution-stream';

const isTerminal = (event: AgentExecutionStreamEnvelope) => (
  event.event === 'run.completed'
  || event.event === 'run.failed'
  || event.event === 'run.canceled'
  || event.event === 'interrupt.created'
);

export default function useBatchedExecutionEvents(initial: AgentExecutionStreamEnvelope[] = []) {
  const [events, setEvents] = useState<AgentExecutionStreamEnvelope[]>(initial);
  const pending = useRef<AgentExecutionStreamEnvelope[]>([]);
  const frame = useRef<number | null>(null);

  const flush = useCallback(() => {
    if (frame.current !== null && typeof window !== 'undefined') window.cancelAnimationFrame(frame.current);
    frame.current = null;
    if (pending.current.length === 0) return;
    const next = pending.current;
    pending.current = [];
    setEvents((current) => [...current, ...next]);
  }, []);

  const append = useCallback((event: AgentExecutionStreamEnvelope) => {
    pending.current.push(event);
    if (isTerminal(event)) {
      flush();
      return;
    }
    if (frame.current === null && typeof window !== 'undefined') {
      frame.current = window.requestAnimationFrame(flush);
    }
  }, [flush]);

  const reset = useCallback((next: AgentExecutionStreamEnvelope[] = []) => {
    pending.current = [];
    if (frame.current !== null && typeof window !== 'undefined') window.cancelAnimationFrame(frame.current);
    frame.current = null;
    setEvents(next);
  }, []);

  useEffect(() => () => {
    if (frame.current !== null) window.cancelAnimationFrame(frame.current);
  }, []);

  return { events, append, reset, setEvents };
}
