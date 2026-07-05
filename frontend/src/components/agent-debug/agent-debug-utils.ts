export const formatTraceError = (error: unknown) => {
  if (!error) return null;
  if (typeof error === 'string') return error;
  if (typeof error === 'object') {
    const err = error as Record<string, any>;
    return String(err.message || err.code || err.type || JSON.stringify(err));
  }
  return String(error);
};
