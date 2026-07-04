import prettyMilliseconds from 'pretty-ms';

type FormatDurationOptions = {
  showZero?: boolean;
};

export const formatDurationMs = (
  value?: number | string | null,
  options: FormatDurationOptions = {},
) => {
  const durationMs = Number(value);
  if (!Number.isFinite(durationMs) || durationMs < 0) return null;
  if (durationMs === 0 && !options.showZero) return null;

  return prettyMilliseconds(durationMs, {
    millisecondsDecimalDigits: 0,
    secondsDecimalDigits: durationMs < 10_000 ? 1 : 0,
    unitCount: 2,
  });
};
