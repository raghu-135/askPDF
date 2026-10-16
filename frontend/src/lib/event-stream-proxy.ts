export const isEventStreamContentType = (contentType: string | null | undefined) => (
  Boolean(contentType?.toLowerCase().includes('text/event-stream'))
);

export const eventStreamProxyHeaders = (contentType: string | null | undefined): Record<string, string> => ({
  'Content-Type': contentType || 'text/event-stream',
  'Cache-Control': 'no-cache, no-transform',
  Connection: 'keep-alive',
  'X-Accel-Buffering': 'no',
});
