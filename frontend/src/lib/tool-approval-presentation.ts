/** Plain-language copy for tool-approval cards (display names from prompt-tool catalog). */

export type ToolDisplayCatalog = Record<string, { display_name?: string; description?: string }>;

function humanizeToolName(toolName: string): string {
  return String(toolName || 'tool')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

export function toolApprovalDisplayName(
  toolName: string,
  catalog?: ToolDisplayCatalog,
): string {
  const entry = catalog?.[toolName];
  if (entry?.display_name?.trim()) return entry.display_name.trim();
  return humanizeToolName(toolName);
}

function primaryArgumentPhrase(argumentsJson: Record<string, unknown> | undefined): string | null {
  const args = argumentsJson || {};
  const preferredKeys = ['query', 'q', 'question', 'search', 'url', 'symbol', 'topic'];
  for (const key of preferredKeys) {
    const value = args[key];
    if (typeof value === 'string' && value.trim()) return value.trim();
  }
  const firstString = Object.values(args).find((value) => typeof value === 'string' && value.trim());
  return typeof firstString === 'string' ? firstString.trim() : null;
}

export function toolApprovalSummaryLine(
  toolName: string,
  argumentsJson: Record<string, unknown> | undefined,
  catalog?: ToolDisplayCatalog,
): string {
  const display = toolApprovalDisplayName(toolName, catalog);
  const phrase = primaryArgumentPhrase(argumentsJson);
  const description = catalog?.[toolName]?.description?.trim();
  if (phrase) {
    const lower = display.toLowerCase();
    if (lower.includes('search') || lower.includes('internet') || lower.includes('web')) {
      return `Search the web for: ${phrase}`;
    }
    if (lower.includes('wikipedia') || lower.includes('wiki')) {
      return `Look up on Wikipedia: ${phrase}`;
    }
    return `${display}: ${phrase}`;
  }
  if (description) return description;
  return `${display} with the proposed parameters below.`;
}