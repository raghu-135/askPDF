import assert from 'node:assert/strict';
import test from 'node:test';

import { shouldClearCurrentOnSourceKeyChange } from '../src/lib/player-source-key.ts';

test('shouldClearCurrentOnSourceKeyChange skips the first source key mount', () => {
  assert.equal(shouldClearCurrentOnSourceKeyChange(null, 'pdf:hash-a'), false);
});

test('shouldClearCurrentOnSourceKeyChange clears when switching documents', () => {
  assert.equal(shouldClearCurrentOnSourceKeyChange('pdf:hash-a', 'pdf:hash-b'), true);
  assert.equal(shouldClearCurrentOnSourceKeyChange('pdf:hash-a', 'pdf:hash-a'), false);
});
