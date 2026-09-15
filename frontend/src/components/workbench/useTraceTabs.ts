import { useCallback, useEffect, useRef, useState } from 'react';
import type { ChatTraceDescriptor } from '../ChatInterface';
import { closeTraceTab, canOpenTraceTab, isValidTraceId, pendingTraceTabId, upsertTraceTab } from '../../lib/trace-tabs';
import type { TraceRunTab } from './TraceWorkspace';

export default function useTraceTabs() {
  const [traceTabs, setTraceTabs] = useState<TraceRunTab[]>([]);
  const [activeTraceId, setActiveTraceId] = useState<string | null>(null);
  const pendingTraceUpdatesRef = useRef(new Map<string, ChatTraceDescriptor>());
  const knownTabIdsRef = useRef(new Set<string>());
  const flushTimerRef = useRef<number | null>(null);

  const flushPendingTraceUpdates = useCallback(() => {
    flushTimerRef.current = null;
    const pending = Array.from(pendingTraceUpdatesRef.current.values());
    pendingTraceUpdatesRef.current.clear();
    if (pending.length === 0) return;
    pending.forEach((trace) => knownTabIdsRef.current.add(trace.id));
    setTraceTabs((current) => pending.reduce((tabs, trace) => upsertTraceTab(tabs, trace), current));
  }, []);

  const scheduleTraceFlush = useCallback(() => {
    if (flushTimerRef.current !== null) return;
    flushTimerRef.current = window.setTimeout(flushPendingTraceUpdates, 120);
  }, [flushPendingTraceUpdates]);

  const openTrace = useCallback((trace: ChatTraceDescriptor) => {
    if (!canOpenTraceTab(trace?.id)) return;
    if (isValidTraceId(trace.id) && trace.messageId) {
      const pendingId = pendingTraceTabId(trace.messageId);
      pendingTraceUpdatesRef.current.delete(pendingId);
      knownTabIdsRef.current.delete(pendingId);
    }
    if (trace.activate !== false) setActiveTraceId(trace.id);
    if (!trace.running) {
      pendingTraceUpdatesRef.current.delete(trace.id);
      knownTabIdsRef.current.add(trace.id);
      setTraceTabs((current) => upsertTraceTab(current, trace));
      return;
    }
    const isFirstSnapshot = !knownTabIdsRef.current.has(trace.id);
    pendingTraceUpdatesRef.current.set(trace.id, trace);
    if (isFirstSnapshot) {
      flushPendingTraceUpdates();
      return;
    }
    scheduleTraceFlush();
  }, [flushPendingTraceUpdates, scheduleTraceFlush]);

  const closeTrace = useCallback((runId: string) => {
    pendingTraceUpdatesRef.current.delete(runId);
    knownTabIdsRef.current.delete(runId);
    setTraceTabs((current) => {
      const result = closeTraceTab(current, activeTraceId, runId);
      setActiveTraceId(result.activeId);
      return result.tabs;
    });
  }, [activeTraceId]);

  const clearTraces = useCallback(() => {
    pendingTraceUpdatesRef.current.clear();
    knownTabIdsRef.current.clear();
    if (flushTimerRef.current !== null) {
      window.clearTimeout(flushTimerRef.current);
      flushTimerRef.current = null;
    }
    setTraceTabs([]);
    setActiveTraceId(null);
  }, []);

  useEffect(() => () => {
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
