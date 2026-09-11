import { useCallback, useEffect, useRef, useState } from 'react';
import type { ChatTraceDescriptor } from '../ChatInterface';
import { streamAgentRunEvents, type BuilderTestStreamEnvelope } from '../../lib/api';
import { AGENT_SSE_RECONNECT_INTERVAL_MS } from '../../lib/agent-ui-config';
import { closeTraceTab, mergeTraceTab, upsertTraceTab } from '../../lib/trace-tabs';
import { buildLiveTraceView } from '../agent-debug/agent-trace-projection';
import type { TraceRunTab } from './TraceWorkspace';

const TERMINAL_RUN_EVENTS = new Set(['run.completed', 'run.failed', 'run.cancelled', 'run.clarification']);
type LiveTraceStream = {
  controller: AbortController;
  events: BuilderTestStreamEnvelope[];
  afterSequence: number;
  terminal: boolean;
};

function waitForReconnect(signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal.aborted) {
      resolve();
      return;
    }
    const timer = window.setTimeout(resolve, AGENT_SSE_RECONNECT_INTERVAL_MS);
    signal.addEventListener('abort', () => {
      window.clearTimeout(timer);
      resolve();
    }, { once: true });
  });
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}

export default function useTraceTabs() {
  const [traceTabs, setTraceTabs] = useState<TraceRunTab[]>([]);
  const [activeTraceId, setActiveTraceId] = useState<string | null>(null);
  const pendingTraceUpdatesRef = useRef(new Map<string, ChatTraceDescriptor>());
  const traceDescriptorsRef = useRef(new Map<string, ChatTraceDescriptor>());
  const liveTraceStreamsRef = useRef(new Map<string, LiveTraceStream>());
  const flushTimerRef = useRef<number | null>(null);

  const flushPendingTraceUpdates = useCallback(() => {
    flushTimerRef.current = null;
    const pending = Array.from(pendingTraceUpdatesRef.current.values());
    pendingTraceUpdatesRef.current.clear();
    if (pending.length === 0) return;
    setTraceTabs((current) => pending.reduce((tabs, trace) => upsertTraceTab(tabs, trace), current));
  }, []);

  const scheduleTraceFlush = useCallback(() => {
    if (flushTimerRef.current !== null) return;
    flushTimerRef.current = window.setTimeout(flushPendingTraceUpdates, 120);
  }, [flushPendingTraceUpdates]);

  const queueTraceUpdate = useCallback((trace: ChatTraceDescriptor, immediate = false) => {
    const merged = mergeTraceTab(traceDescriptorsRef.current.get(trace.id), trace);
    traceDescriptorsRef.current.set(trace.id, merged);
    if (immediate) {
      pendingTraceUpdatesRef.current.delete(trace.id);
      setTraceTabs((current) => upsertTraceTab(current, merged));
      return;
    }
    pendingTraceUpdatesRef.current.set(trace.id, merged);
    scheduleTraceFlush();
  }, [scheduleTraceFlush]);

  const stopLiveTraceStream = useCallback((runId: string) => {
    const stream = liveTraceStreamsRef.current.get(runId);
    if (!stream) return;
    stream.controller.abort();
    liveTraceStreamsRef.current.delete(runId);
  }, []);

  const startLiveTraceStream = useCallback((trace: ChatTraceDescriptor) => {
    if (trace.liveEventSource !== 'agent_run_events' || !trace.threadId || liveTraceStreamsRef.current.has(trace.id)) return;
    const stream: LiveTraceStream = {
      controller: new AbortController(),
      events: [],
      afterSequence: 0,
      terminal: false,
    };
    liveTraceStreamsRef.current.set(trace.id, stream);

    const consume = async () => {
      while (!stream.controller.signal.aborted && !stream.terminal) {
        try {
          await streamAgentRunEvents(
            trace.id,
            trace.threadId!,
            stream.afterSequence,
            (event) => {
              const sequence = Number(event.data.sequence || 0);
              if (sequence > 0 && sequence <= stream.afterSequence) return;
              stream.afterSequence = Math.max(stream.afterSequence, sequence);
              stream.events = [...stream.events, event];
              const terminal = event.data.terminal === true || TERMINAL_RUN_EVENTS.has(event.event);
              stream.terminal = terminal;
              const current = traceDescriptorsRef.current.get(trace.id) || trace;
              queueTraceUpdate({
                ...current,
                status: terminal ? event.event.slice(4) : 'running',
                liveTraceView: buildLiveTraceView(stream.events),
                running: !terminal,
              }, terminal);
            },
            stream.controller.signal,
          );
        } catch (error) {
          if (!isAbortError(error) && !stream.controller.signal.aborted) {
            await waitForReconnect(stream.controller.signal);
          }
          continue;
        }
        if (!stream.terminal && !stream.controller.signal.aborted) {
          await waitForReconnect(stream.controller.signal);
        }
      }
      if (liveTraceStreamsRef.current.get(trace.id) === stream) {
        liveTraceStreamsRef.current.delete(trace.id);
      }
    };
    void consume();
  }, [queueTraceUpdate]);

  const openTrace = useCallback((trace: ChatTraceDescriptor) => {
    setActiveTraceId(trace.id);
    const merged = mergeTraceTab(traceDescriptorsRef.current.get(trace.id), trace);
    traceDescriptorsRef.current.set(trace.id, merged);
    if (merged.liveEventSource === 'agent_run_events' && merged.running) {
      startLiveTraceStream(merged);
    } else if (!merged.running) {
      stopLiveTraceStream(merged.id);
    }
    if (trace.running) {
      pendingTraceUpdatesRef.current.set(trace.id, merged);
      scheduleTraceFlush();
      return;
    }
    pendingTraceUpdatesRef.current.delete(trace.id);
    setTraceTabs((current) => upsertTraceTab(current, merged));
  }, [scheduleTraceFlush, startLiveTraceStream, stopLiveTraceStream]);

  const closeTrace = useCallback((runId: string) => {
    stopLiveTraceStream(runId);
    pendingTraceUpdatesRef.current.delete(runId);
    traceDescriptorsRef.current.delete(runId);
    setTraceTabs((current) => {
      const result = closeTraceTab(current, activeTraceId, runId);
      setActiveTraceId(result.activeId);
      return result.tabs;
    });
  }, [activeTraceId, stopLiveTraceStream]);

  const clearTraces = useCallback(() => {
    liveTraceStreamsRef.current.forEach((stream) => stream.controller.abort());
    liveTraceStreamsRef.current.clear();
    pendingTraceUpdatesRef.current.clear();
    traceDescriptorsRef.current.clear();
    if (flushTimerRef.current !== null) {
      window.clearTimeout(flushTimerRef.current);
      flushTimerRef.current = null;
    }
    setTraceTabs([]);
    setActiveTraceId(null);
  }, []);

  useEffect(() => () => {
    liveTraceStreamsRef.current.forEach((stream) => stream.controller.abort());
    liveTraceStreamsRef.current.clear();
    if (flushTimerRef.current !== null) {
      window.clearTimeout(flushTimerRef.current);
    }
  }, []);

  return {
    traceTabs,
    activeTraceId,
    setActiveTraceId,
    openTrace,
    closeTrace,
    clearTraces,
  };
}
