/** Agent run statuses that should show the trace tab "Live" chip. */
export function isAgentRunActivelyStreaming(status: string | null | undefined): boolean {
  return String(status || '').trim() === 'running';
}

export function isTraceTabLive(tab: {
  status?: string;
  running?: boolean;
  runDetails?: { status?: string };
}): boolean {
  const status = tab.runDetails?.status ?? tab.status;
  if (status) return isAgentRunActivelyStreaming(status);
  return Boolean(tab.running);
}
