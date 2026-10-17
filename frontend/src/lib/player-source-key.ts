/** Whether switching PDF/chat playback sources should clear the active sentence id. */
export function shouldClearCurrentOnSourceKeyChange(
  previousKey: string | null,
  nextKey: string,
): boolean {
  return previousKey !== null && previousKey !== nextKey;
}
