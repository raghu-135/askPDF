export function activityTimestamp(item: {
  last_activity_at?: string | null;
  updated_at?: string | null;
  created_at?: string | null;
}): number {
  const value = item.last_activity_at || item.updated_at || item.created_at;
  if (!value) return 0;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export function sortByRecentActivity<T extends {
  last_activity_at?: string | null;
  updated_at?: string | null;
  created_at?: string | null;
}>(items: readonly T[]): T[] {
  return [...items].sort((left, right) => activityTimestamp(right) - activityTimestamp(left));
}
