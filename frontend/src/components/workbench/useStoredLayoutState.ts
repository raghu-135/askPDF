import { useEffect, useMemo, useRef, useState } from 'react';
import { readStoredLayoutState, writeStoredLayoutState } from '../../lib/stored-layout-state';

export default function useStoredLayoutState<T>(
  storageKey: string,
  fallback: T,
  normalize: (value?: unknown) => T,
) {
  const normalizedFallback = useMemo(() => normalize(fallback), [fallback, normalize]);
  const [state, setState] = useState<T>(() => {
    if (typeof window === 'undefined') return normalizedFallback;
    return readStoredLayoutState(
      window.localStorage.getItem.bind(window.localStorage),
      storageKey,
      normalizedFallback,
      normalize,
    );
  });
  const hydratedStorageKeyRef = useRef(storageKey);

  useEffect(() => {
    if (hydratedStorageKeyRef.current === storageKey) return;
    hydratedStorageKeyRef.current = storageKey;
    if (typeof window === 'undefined') {
      setState(normalizedFallback);
      return;
    }
    setState(readStoredLayoutState(
      window.localStorage.getItem.bind(window.localStorage),
      storageKey,
      normalizedFallback,
      normalize,
    ));
  }, [normalizedFallback, normalize, storageKey]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      writeStoredLayoutState(window.localStorage.setItem.bind(window.localStorage), storageKey, state);
    } catch {
      // Persistence is best effort.
    }
  }, [state, storageKey]);

  return [state, setState] as const;
}
