import { useCallback, useEffect, useState } from 'react';

export type WebSearchMode = 'off' | 'ask' | 'on';

const STORAGE_KEY = 'last_web_search_mode';
const LEGACY_KEY = 'last_use_web_search';
const EVENT_NAME = 'askpdf:web-search-mode';

export const nextWebSearchMode = (mode: WebSearchMode): WebSearchMode =>
  mode === 'off' ? 'ask' : mode === 'ask' ? 'on' : 'off';

export const readWebSearchMode = (): WebSearchMode => {
  if (typeof window === 'undefined') return 'off';
  const saved = window.localStorage.getItem(STORAGE_KEY);
  if (saved === 'off' || saved === 'ask' || saved === 'on') return saved;
  const legacy = window.localStorage.getItem(LEGACY_KEY);
  return legacy === '1' ? 'on' : 'off';
};

export const persistWebSearchMode = (mode: WebSearchMode) => {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(STORAGE_KEY, mode);
  window.localStorage.setItem(LEGACY_KEY, mode === 'off' ? '0' : '1');
  window.dispatchEvent(new CustomEvent<WebSearchMode>(EVENT_NAME, { detail: mode }));
};

export function useWebSearchMode(initialMode?: WebSearchMode) {
  const [mode, setModeState] = useState<WebSearchMode>(() => initialMode ?? readWebSearchMode());
  useEffect(() => {
    const sync = (event: Event) => {
      const detail = (event as CustomEvent<WebSearchMode>).detail;
      setModeState(detail || readWebSearchMode());
    };
    const storage = (event: StorageEvent) => {
      if (event.key === STORAGE_KEY || event.key === LEGACY_KEY) setModeState(readWebSearchMode());
    };
    window.addEventListener(EVENT_NAME, sync);
    window.addEventListener('storage', storage);
    setModeState(readWebSearchMode());
    return () => {
      window.removeEventListener(EVENT_NAME, sync);
      window.removeEventListener('storage', storage);
    };
  }, []);
  const setMode = useCallback((next: WebSearchMode) => {
    setModeState(next);
    persistWebSearchMode(next);
  }, []);
  return { mode, setMode };
}
