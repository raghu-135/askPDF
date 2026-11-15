import { useCallback, useState } from 'react';
import {
  closeInstanceTab,
  type IdentifiedInstanceTab,
} from '../../lib/instance-tabs';

export default function useInstanceTabs<T extends IdentifiedInstanceTab>() {
  const [tabs, setTabs] = useState<T[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  const close = useCallback((id: string) => {
    setTabs((current) => {
      const result = closeInstanceTab(current, activeId, id);
      setActiveId(result.activeId);
      return result.tabs;
    });
  }, [activeId]);

  const clear = useCallback(() => {
    setTabs([]);
    setActiveId(null);
  }, []);

  return {
    tabs,
    setTabs,
    activeId,
    setActiveId,
    close,
    clear,
  };
}