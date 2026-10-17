export type CitationDisplaySource = {
  kind?: string;
  label?: string;
  file_name?: string;
  page?: number | string;
  source_id?: string;
  origins?: Array<{ source_id?: string } | Record<string, unknown>>;
};

function pagePhrase(page: number | string | undefined): string | null {
  if (page === undefined || page === null || page === '') return null;
  return `page ${page}`;
}

export function citationLabelFromSource(source: CitationDisplaySource): string | null {
  const name = String(source.file_name || source.label || '').trim();
  const page = pagePhrase(source.page);
  if (name && page) return `${name}, ${page}`;
  if (name) return name;
  if (page) return `document, ${page}`;
  return null;
}

export function buildCitationDisplayMap(sources: CitationDisplaySource[] | undefined): Map<string, string> {
  const map = new Map<string, string>();
  (sources || []).forEach((source) => {
    const label = citationLabelFromSource(source);
    if (!label) return;
    const ids = [
      source.source_id,
      ...(source.origins || []).map((origin) => {
        if (origin && typeof origin === 'object' && 'source_id' in origin) {
          const value = (origin as { source_id?: string }).source_id;
          return typeof value === 'string' ? value : undefined;
        }
        return undefined;
      }),
    ].filter(Boolean) as string[];
    ids.forEach((id) => {
      if (!map.has(id)) map.set(id, label);
    });
  });
  return map;
}

function docIdFallbackLabel(sourceId: string): string | null {
  if (!sourceId.startsWith('doc:')) return null;
  const parts = sourceId.split(':');
  if (parts.length < 3) return null;
  const locator = parts.slice(2).join(':');
  if (/^\d+$/.test(locator)) return `document, page ${locator}`;
  return `document, section ${locator}`;
}

export function rewriteDocCitationTokens(content: string, displayMap: Map<string, string>): string {
  if (!content.trim() || !displayMap.size) return content;
  let result = content;
  displayMap.forEach((label, id) => {
    const escaped = id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const replacement = `${label} (\`${id}\`)`;
    result = result.replace(new RegExp(`\`${escaped}\``, 'g'), replacement);
    result = result.replace(new RegExp(`(?<![\`\\w])${escaped}(?![\`\\w])`, 'g'), replacement);
  });
  const docPattern = /doc:[^:\s`]+:[^\s`,.)]+/g;
  result = result.replace(docPattern, (match) => {
    if (displayMap.has(match)) return `${displayMap.get(match)!} (\`${match}\`)`;
    const fallback = docIdFallbackLabel(match);
    return fallback ? `${fallback} (\`${match}\`)` : match;
  });
  return result;
}
