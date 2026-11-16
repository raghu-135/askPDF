export type ClarificationChoice = {
  text: string;
};

export const clarificationChoiceText = (choice: unknown): string => {
  if (typeof choice === 'string') return choice;
  if (choice && typeof choice === 'object') {
    const candidate = choice as { text?: unknown; label?: unknown; title?: unknown; question?: unknown };
    for (const value of [candidate.text, candidate.label, candidate.title, candidate.question]) {
      if (typeof value === 'string') return value;
    }
  }
  return '';
};

export function clarificationChoicesFromPayload(
  payload: { clarification_options?: unknown; clarification?: unknown } | null | undefined,
): ClarificationChoice[] {
  const collected: unknown[] = [];
  const direct = payload?.clarification_options;
  if (Array.isArray(direct)) {
    collected.push(...direct);
  }
  const nested = payload?.clarification;
  if (nested && typeof nested === 'object') {
    const options = (nested as { options?: unknown }).options;
    if (Array.isArray(options)) {
      collected.push(...options);
    }
  }
  return collected
    .map((choice) => clarificationChoiceText(choice).trim())
    .filter(Boolean)
    .map((text) => ({ text }));
}