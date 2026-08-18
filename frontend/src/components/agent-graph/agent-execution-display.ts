export const compactExecutionText = (value: unknown, maxCharacters = 240): string => {
  const text = typeof value === 'string'
    ? value
    : value === undefined || value === null
      ? ''
      : JSON.stringify(value);
  const plain = text
    .replace(/\*\*|__/g, '')
    .replace(/`+/g, '')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/\s+/g, ' ')
    .trim();
  if (plain.length <= maxCharacters) return plain;
  return `${plain.slice(0, Math.max(1, maxCharacters - 1)).trimEnd()}…`;
};
