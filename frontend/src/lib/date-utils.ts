/**
 * Date formatting utilities.
 */

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
