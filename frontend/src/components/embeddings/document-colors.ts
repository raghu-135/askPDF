const HUE_OFFSET_DEGREES = 210;

function uniqueFileHashes(fileHashes: readonly string[]): string[] {
  const unique: string[] = [];
  const seen = new Set<string>();
  for (const hash of fileHashes) {
    if (!hash || seen.has(hash)) continue;
    seen.add(hash);
    unique.push(hash);
  }
  return unique;
}

function hslToHex(hue: number, saturation: number, lightness: number): string {
  const h = ((hue % 360) + 360) % 360;
  const s = saturation / 100;
  const l = lightness / 100;
  const chroma = (1 - Math.abs(2 * l - 1)) * s;
  const x = chroma * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - chroma / 2;
  let r = 0;
  let g = 0;
  let b = 0;
  if (h < 60) {
    r = chroma;
    g = x;
  } else if (h < 120) {
    r = x;
    g = chroma;
  } else if (h < 180) {
    g = chroma;
    b = x;
  } else if (h < 240) {
    g = x;
    b = chroma;
  } else if (h < 300) {
    r = x;
    b = chroma;
  } else {
    r = chroma;
    b = x;
  }
  const toHex = (channel: number) => Math.round((channel + m) * 255).toString(16).padStart(2, '0');
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

export function documentFill(index: number, count: number, mode: 'light' | 'dark'): string {
  const total = Math.max(count, 1);
  const hue = ((index * 360) / total + HUE_OFFSET_DEGREES) % 360;
  const saturation = mode === 'dark' ? 72 : 68;
  const baseLightness = mode === 'dark' ? 62 : 38;
  const lightnessShift = total > 8 && index % 2 === 1
    ? (mode === 'dark' ? -7 : 7)
    : 0;
  return hslToHex(hue, saturation, baseLightness + lightnessShift);
}

export function assignDocumentColors(
  fileHashes: readonly string[],
  mode: 'light' | 'dark',
): Record<string, string> {
  const unique = uniqueFileHashes(fileHashes);
  return Object.fromEntries(
    unique.map((hash, index) => [hash, documentFill(index, unique.length, mode)]),
  );
}

export function truncateLabel(value: string, maxChars: number): string {
  const trimmed = value.trim();
  if (trimmed.length <= maxChars) return trimmed;
  return `${trimmed.slice(0, Math.max(1, maxChars - 1))}…`;
}

export function chunkGraphLabel(
  fileName: string | null | undefined,
  fileHash: string,
  chunkId: number | string | null | undefined,
): string {
  const name = truncateLabel(fileName || fileHash.slice(0, 8), 18);
  const chunk = chunkId == null ? '?' : String(chunkId);
  return `${name} · c${chunk}`;
}
