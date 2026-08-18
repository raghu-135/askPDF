export const flexTruncateSx = {
  minWidth: 0,
  maxWidth: '100%',
} as const;

export const singleLineTruncateSx = {
  minWidth: 0,
  maxWidth: '100%',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
} as const;
