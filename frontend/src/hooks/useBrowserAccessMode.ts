import { useCallback, useEffect, useState } from 'react';

export type BrowserAccessMode = 'off' | 'ask' | 'on';

const STORAGE_KEY = 'last_browser_access_mode';
const EVENT_NAME = 'askpdf:browser-access-mode';

export const nextBrowserAccessMode = (mode: BrowserAccessMode): BrowserAccessMode =>
  mode === 'off' ? 'ask' : mode === 'ask' ? 'on' : 'off';

export const readBrowserAccessMode = (): BrowserAccessMode => {
  if (typeof window === 'undefined') return 'off';
  const saved = window.localStorage.getItem(STORAGE_KEY);
  if (saved === 'off' || saved === 'ask' || saved === 'on') return saved;
  return 'off';
};

export const persistBrowserAccessMode = (mode: BrowserAccessMode) => {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(STORAGE_KEY, mode);
  window.dispatchEvent(new CustomEvent<BrowserAccessMode>(EVENT_NAME, { detail: mode }));
};

export function useBrowserAccessMode(initialMode?: BrowserAccessMode) {
  const [mode, setModeState] = useState<BrowserAccessMode>(() => initialMode ?? readBrowserAccessMode());
  useEffect(() => {
    const sync = (event: Event) => {
      const detail = (event as CustomEvent<BrowserAccessMode>).detail;
      setModeState(detail || readBrowserAccessMode());
    };
    const storage = (event: StorageEvent) => {
      if (event.key === STORAGE_KEY) setModeState(readBrowserAccessMode());
    };
    window.addEventListener(EVENT_NAME, sync);
    window.addEventListener('storage', storage);
    setModeState(readBrowserAccessMode());
    return () => {
      window.removeEventListener(EVENT_NAME, sync);
      window.removeEventListener('storage', storage);
    };
  }, []);
  const setMode = useCallback((next: BrowserAccessMode) => {
    setModeState(next);
    persistBrowserAccessMode(next);
  }, []);
  return { mode, setMode };
}