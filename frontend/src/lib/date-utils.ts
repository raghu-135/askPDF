/**
 * Date formatting utilities.
 */

export interface BrowserRuntimeContext {
  client_timezone: string;
  client_locale: string;
  client_now_iso: string;
}

export function getBrowserRuntimeContext(): BrowserRuntimeContext {
  const fallbackTimezone = "UTC";
  let timezone = fallbackTimezone;
  try {
    timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || fallbackTimezone;
  } catch {
    timezone = fallbackTimezone;
  }

  return {
    client_timezone: timezone,
    client_locale: typeof navigator !== "undefined" ? navigator.language : "",
    client_now_iso: new Date().toISOString(),
  };
}

/**
 * Formats a date string into a human-readable label (Today, Yesterday, X days ago, or locale date).
 * @param dateStr - The date string to format.
 * @returns A formatted date string.
 */
export const formatDate = (dateStr: string): string => {
  const date = new Date(dateStr);
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));
  if (days === 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days} days ago`;
  return date.toLocaleDateString();
};
