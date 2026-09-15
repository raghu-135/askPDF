export const formatAgentTraceIdentity = ({
  workflowId,
  route,
}: {
  workflowId?: string | null;
  route?: string | null;
}) => {
  const workflow = String(workflowId || '').trim() || 'agent';
  const routeLabel = String(route || '').trim();
  return routeLabel ? `${workflow} · ${routeLabel}` : workflow;
};

export const formatOpenTraceControlLabel = ({
  running,
  canceling,
  workflowId,
  route,
}: {
  running?: boolean;
  canceling?: boolean;
  workflowId?: string | null;
  route?: string | null;
}) => {
  if (canceling) return 'Stopping after current step…';
  const identity = formatAgentTraceIdentity({ workflowId, route });
  return running ? `Open live trace · ${identity}` : `Open trace · ${identity}`;
};
