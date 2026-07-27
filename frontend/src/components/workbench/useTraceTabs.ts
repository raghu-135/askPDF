import { useCallback, useState } from 'react';
import type { ChatTraceDescriptor } from '../ChatInterface';
import { closeTraceTab, upsertTraceTab } from '../../lib/trace-tabs';
import type { TraceRunTab } from './TraceWorkspace';

export default function useTraceTabs() {
  const [traceTabs, setTraceTabs] = useState<TraceRunTab[]>([]);
  const [activeTraceId, setActiveTraceId] = useState<string | null>(null);

  const openTrace = useCallback((trace: ChatTraceDescriptor) => {
    setTraceTabs((current) => upsertTraceTab(current, trace));
    setActiveTraceId(trace.id);
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
