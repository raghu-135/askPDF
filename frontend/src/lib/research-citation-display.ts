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
  if (/^c-\d+$/i.test(locator)) return `document, section ${locator}`;
  return `document, section ${locator}`;
}

const COMBINED_PAREN_CITATION_PATTERN = /\(([^)]+\.pdf,\s*page\s+\d+)\s+(doc:[^\s)]+)\)/gi;

export function rewriteDocCitationTokens(content: string, displayMap: Map<string, string>): string {
  if (!content.trim()) return content;
  let result = content;
  result = result.replace(COMBINED_PAREN_CITATION_PATTERN, (_match, humanLabel: string, sourceId: string) => {
    const id = String(sourceId || '').trim();
    const label = (displayMap.size ? displayMap.get(id) : undefined) || String(humanLabel || '').trim();
    return `${label} (\`${id}\`)`;
  });
  if (!displayMap.size) return result;
  displayMap.forEach((label, id) => {
    const escaped = id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const replacement = `${label} (\`${id}\`)`;
    result = result.replace(new RegExp(`\`${escaped}\``, 'g'), replacement);
    result = result.replace(new RegExp(`(?<![\`\\w])${escaped}(?![\`\\w])`, 'g'), replacement);
  });
  const docPattern = /doc:[^:\s`]+:[^\s`,.)]+/gi;
  result = result.replace(docPattern, (match) => {
    if (displayMap.has(match)) return `${displayMap.get(match)!} (\`${match}\`)`;
    const fallback = docIdFallbackLabel(match);
    return fallback ? `${fallback} (\`${match}\`)` : match;
  });
  const parenDocPattern = /\((doc:[^:\s)]+:[^\s),]+)\)/gi;
  result = result.replace(parenDocPattern, (match, sourceId: string) => {
    const id = String(sourceId || '').trim();
    if (!id) return match;
    const label = displayMap.get(id) || docIdFallbackLabel(id);
    return label ? `${label} (\`${id}\`)` : match;
  });
  return result;
}

export function citationSourcesFromTimelineItems(
  items: Array<{
    sources?: Array<{
      kind?: string;
      title?: string;
      file_hash?: string;
      page?: number;
      origins?: Array<{ artifact_id?: string; source_id?: string } | Record<string, unknown>>;
    }>;
  }>,
): CitationDisplaySource[] {
  const mapped: CitationDisplaySource[] = [];
  items.forEach((item) => {
    (item.sources || []).forEach((source) => {
      const fileHash = source.file_hash;
      const locator = source.page ?? 0;
      const docId = fileHash ? `doc:${fileHash}:${locator}` : undefined;
      mapped.push({
        kind: source.kind,
        label: source.title,
        file_name: source.title,
        page: source.page,
        source_id: docId,
        origins: (source.origins || []).map((origin) => ({
          source_id: (origin as { artifact_id?: string }).artifact_id,
        })),
      });
    });
  });
  return mapped;
}
