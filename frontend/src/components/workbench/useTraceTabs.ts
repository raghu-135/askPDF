import { useCallback, useState } from 'react';
import type { ChatTraceDescriptor } from '../ChatInterface';
import { closeTraceTab, canOpenTraceTab, isValidTraceId, pendingTraceTabId, upsertTraceTab } from '../../lib/trace-tabs';
import type { TraceRunTab } from './TraceWorkspace';

export default function useTraceTabs() {
  const [traceTabs, setTraceTabs] = useState<TraceRunTab[]>([]);
  const [activeTraceId, setActiveTraceId] = useState<string | null>(null);

  const openTrace = useCallback((trace: ChatTraceDescriptor) => {
    if (!canOpenTraceTab(trace?.id)) return;
    setTraceTabs((current) => {
      const withoutPending = isValidTraceId(trace.id) && trace.messageId
        ? current.filter((tab) => tab.id !== pendingTraceTabId(trace.messageId as string))
        : current;
      return upsertTraceTab(withoutPending, trace);
    });
    if (trace.activate !== false) setActiveTraceId(trace.id);
  }, []);

  const closeTrace = useCallback((runId: string) => {
    setTraceTabs((current) => {
      const result = closeTraceTab(current, activeTraceId, runId);
      setActiveTraceId(result.activeId);
      return result.tabs;
    });
  }, [activeTraceId]);

  const clearTraces = useCallback(() => {
    setTraceTabs([]);
    setActiveTraceId(null);
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
