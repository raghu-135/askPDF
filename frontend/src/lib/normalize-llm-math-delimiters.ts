const LATEX_COMMAND = /\\[a-zA-Z]+/;

type Segment = { kind: 'text' | 'code'; value: string };

function splitPreservingCodeFences(input: string): Segment[] {
  const segments: Segment[] = [];
  let index = 0;
  while (index < input.length) {
    if (input.startsWith('```', index)) {
      const close = input.indexOf('```', index + 3);
      const end = close === -1 ? input.length : close + 3;
      segments.push({ kind: 'code', value: input.slice(index, end) });
      index = end;
      continue;
    }
    if (input[index] === '`') {
      let cursor = index + 1;
      while (cursor < input.length) {
        if (input[cursor] === '`') {
          segments.push({ kind: 'code', value: input.slice(index, cursor + 1) });
          index = cursor + 1;
          break;
        }
        cursor += 1;
      }
      if (cursor >= input.length) {
        segments.push({ kind: 'code', value: input.slice(index) });
        return segments;
      }
      continue;
    }
    let cursor = index;
    while (cursor < input.length && !input.startsWith('```', cursor) && input[cursor] !== '`') {
      cursor += 1;
    }
    segments.push({ kind: 'text', value: input.slice(index, cursor) });
    index = cursor;
  }
  return segments;
}

function looksLikeMathContent(inner: string): boolean {
  const trimmed = inner.trim();
  if (!trimmed) return false;
  if (/^doc:/i.test(trimmed) || /\.pdf\b/i.test(trimmed)) return false;
  if (/^source:/i.test(trimmed)) return false;
  if (LATEX_COMMAND.test(trimmed)) return true;
  if (/\\frac|\\text|\\sqrt|\\sum|\\int|\\left|\\right/.test(trimmed)) return true;
  const compact = trimmed.replace(/\s+/g, '');
  if (/^[A-Za-z][A-Za-z0-9]*(?:[_^][A-Za-z0-9]+)+$/.test(compact)) return true;
  return false;
}

function escapeCurrencyDollars(segment: string): string {
  return segment.replace(/\$(\d+(?:\.\d+)?)/g, (_, amount: string) => `\\$${amount}`);
}

function replaceBareInlineParenMath(segment: string): string {
  return segment.replace(/\(\s*([^()\n`]+?)\s*\)/g, (match, inner: string) => {
    if (!looksLikeMathContent(inner)) return match;
    return `$${inner.trim()}$`;
  });
}

function transformMathInProse(text: string): string {
  let output = text.replace(/\\\[([\s\S]*?)\\\]/g, (_, inner: string) => `$$\n${inner.trim()}\n$$`);
  output = output.replace(/\\\(([\s\S]*?)\\\)/g, (_, inner: string) => `$${inner.trim()}$`);

  output = output.replace(/^(\s*)\[([\s\S]+?)\]\s*$/gm, (line, lead: string, inner: string) => {
    if (!looksLikeMathContent(inner)) return line;
    return `${lead}$$\n${inner.trim()}\n$$`;
  });

  const pieces = output.split(/(\$\$[\s\S]*?\$\$)/g);
  output = pieces
    .map((piece) => (
      piece.startsWith('$$') && piece.endsWith('$$')
        ? piece
        : escapeCurrencyDollars(replaceBareInlineParenMath(piece))
    ))
    .join('');

  return output;
}

/** Normalize LLM math delimiters to remark-math $ / $$ forms outside code spans. */
export function normalizeLlmMathDelimiters(markdown: string): string {
  return splitPreservingCodeFences(String(markdown ?? ''))
    .map((segment) => (segment.kind === 'code' ? segment.value : transformMathInProse(segment.value)))
    .join('');
}