export const readStoredLayoutState = <T>(
  getItem: (key: string) => string | null,
  storageKey: string,
  fallback: T,
  normalize: (value?: unknown) => T,
): T => {
  try {
    const stored = getItem(storageKey);
    return stored ? normalize(JSON.parse(stored)) : normalize(fallback);
  } catch {
    return normalize(fallback);
  }
};

export const writeStoredLayoutState = <T>(
  setItem: (key: string, value: string) => void,
  storageKey: string,
  state: T,
) => {
  setItem(storageKey, JSON.stringify(state));
};
