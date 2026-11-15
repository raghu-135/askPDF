import { useCallback } from 'react';
import type { ChatTraceDescriptor } from '../ChatInterface';
import { canOpenTraceTab, isValidTraceId, pendingTraceTabId, upsertTraceTab } from '../../lib/trace-tabs';
import type { TraceRunTab } from './TraceWorkspace';
import useInstanceTabs from './useInstanceTabs';

export default function useTraceTabs() {
  const {
    tabs: traceTabs,
    setTabs: setTraceTabs,
    activeId: activeTraceId,
    setActiveId: setActiveTraceId,
    close: closeTrace,
    clear: clearTraces,
  } = useInstanceTabs<TraceRunTab>();

  const openTrace = useCallback((trace: ChatTraceDescriptor) => {
    if (!canOpenTraceTab(trace?.id)) return;
    setTraceTabs((current) => {
      const withoutPending = isValidTraceId(trace.id) && trace.messageId
        ? current.filter((tab) => tab.id !== pendingTraceTabId(trace.messageId as string))
        : current;
      return upsertTraceTab(withoutPending, trace);
    });
    if (trace.activate !== false) setActiveTraceId(trace.id);
  }, [setActiveTraceId, setTraceTabs]);

  return {
    traceTabs,
    activeTraceId,
    setActiveTraceId,
    openTrace,
    closeTrace,
    clearTraces,
  };
}
