import assert from 'node:assert/strict';
import test from 'node:test';

import { activityTimestamp, sortByRecentActivity } from '../src/lib/recency.ts';

test('activity timestamp prefers last activity then updated then created', () => {
  assert.equal(
    activityTimestamp({
      created_at: '2026-09-01T00:00:00.000Z',
      updated_at: '2026-09-02T00:00:00.000Z',
      last_activity_at: '2026-09-03T00:00:00.000Z',
    }),
    Date.parse('2026-09-03T00:00:00.000Z'),
  );
  assert.equal(
    activityTimestamp({ created_at: '2026-09-01T00:00:00.000Z' }),
    Date.parse('2026-09-01T00:00:00.000Z'),
  );
});

test('sortByRecentActivity puts the newest items first', () => {
  const items = [
    { id: 'old', created_at: '2026-09-01T00:00:00.000Z' },
    { id: 'edited', created_at: '2026-08-01T00:00:00.000Z', last_activity_at: '2026-09-10T00:00:00.000Z' },
    { id: 'mid', updated_at: '2026-09-05T00:00:00.000Z' },
  ];
  assert.deepEqual(sortByRecentActivity(items).map((item) => item.id), ['edited', 'mid', 'old']);
});
