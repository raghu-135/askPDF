import type { TraceModelView } from '../components/agent-debug/agent-trace-projection';

export function formatRunProviderCost(value: unknown): string | null {
  const amount = Number(value);
  if (!Number.isFinite(amount) || amount < 0) return null;
  if (amount === 0) return '$0.00';
  if (amount < 0.01) return '<$0.01';
  const precision = amount < 1 ? 4 : 2;
  return `$${amount.toFixed(precision)}`;
}

export function aggregateProviderCostUsd(
  metrics: Record<string, unknown> | undefined,
  models: TraceModelView[],
): number | null {
  const metricValue = Number(metrics?.llm_provider_cost_usd);
  if (Number.isFinite(metricValue) && metricValue >= 0) {
    return metricValue;
  }
  let sum = 0;
  let known = false;
  models.forEach((model) => {
    const usage = model.usage || {};
    const raw = usage.cost ?? usage.provider_cost_usd;
    const value = Number(raw);
    if (!Number.isFinite(value) || value < 0) return;
    sum += value;
    known = true;
  });
  return known ? sum : null;
}
