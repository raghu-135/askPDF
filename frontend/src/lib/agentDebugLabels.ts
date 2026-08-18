const SKIP_REASON_LABELS: Record<string, string> = {
  not_selected_by_plan: 'Not in plan',
  web_search_disabled: 'Web disabled',
  hitl_policy_disabled: 'Review disabled',
};

export const formatSkipReason = (reason?: string | null) => {
  if (!reason) return null;
  return SKIP_REASON_LABELS[reason] || reason.replace(/_/g, ' ');
};
