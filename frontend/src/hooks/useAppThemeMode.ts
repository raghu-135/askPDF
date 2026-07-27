import { useCallback, useEffect, useMemo, useState } from 'react';

export type AppThemeMode = 'system' | 'light' | 'dark';

export const APP_THEME_MODE_STORAGE_KEY = 'askpdf.themeMode.v1';
const APP_THEME_MODE_CHANGED_EVENT = 'askpdf-theme-mode-changed';

const readStoredThemeMode = (): AppThemeMode => {
  if (typeof window === 'undefined') return 'system';
  const stored = window.localStorage.getItem(APP_THEME_MODE_STORAGE_KEY);
  return stored === 'light' || stored === 'dark' || stored === 'system' ? stored : 'system';
};

const systemPrefersDark = () => (
  typeof window !== 'undefined' && window.matchMedia
    ? window.matchMedia('(prefers-color-scheme: dark)').matches
    : false
);

export function useAppThemeMode() {
  const [mode, setMode] = useState<AppThemeMode>(() => readStoredThemeMode());
  const [systemDark, setSystemDark] = useState<boolean>(() => systemPrefersDark());
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setMode(readStoredThemeMode());
    setSystemDark(systemPrefersDark());
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (event: MediaQueryListEvent) => setSystemDark(event.matches);
    media.addEventListener('change', handler);
    return () => media.removeEventListener('change', handler);
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const syncStoredMode = () => setMode(readStoredThemeMode());
    const handleStorage = (event: StorageEvent) => {
      if (event.key === APP_THEME_MODE_STORAGE_KEY) syncStoredMode();
    };
    window.addEventListener('storage', handleStorage);
    window.addEventListener(APP_THEME_MODE_CHANGED_EVENT, syncStoredMode);
    return () => {
      window.removeEventListener('storage', handleStorage);
      window.removeEventListener(APP_THEME_MODE_CHANGED_EVENT, syncStoredMode);
    };
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined' || !hydrated) return;
    window.localStorage.setItem(APP_THEME_MODE_STORAGE_KEY, mode);
  }, [hydrated, mode]);

  const darkMode = mode === 'system' ? systemDark : mode === 'dark';
  const toggleDarkMode = useCallback(() => {
    setMode((current) => {
      const currentDarkMode = current === 'system' ? systemPrefersDark() : current === 'dark';
      return currentDarkMode ? 'light' : 'dark';
    });
  }, []);

  const setSyncedMode = useCallback((nextMode: AppThemeMode) => {
    setMode(nextMode);
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(APP_THEME_MODE_STORAGE_KEY, nextMode);
      window.dispatchEvent(new Event(APP_THEME_MODE_CHANGED_EVENT));
    }
  }, []);

  return useMemo(() => ({
    mode,
    setMode: setSyncedMode,
    darkMode,
    toggleDarkMode,
    hydrated,
  }), [darkMode, hydrated, mode, setSyncedMode, toggleDarkMode]);
}
