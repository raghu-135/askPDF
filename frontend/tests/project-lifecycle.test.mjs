import assert from 'node:assert/strict';
import test from 'node:test';

import {
  defaultProjectCloneName,
  projectDeletionConfirmed,
} from '../src/lib/project-lifecycle.ts';

test('project clone names use the source project name', () => {
  assert.equal(defaultProjectCloneName('Research'), 'Research (Copy)');
});

test('project deletion requires an exact case-sensitive name', () => {
  assert.equal(projectDeletionConfirmed('Research', 'Research'), true);
  assert.equal(projectDeletionConfirmed('research', 'Research'), false);
  assert.equal(projectDeletionConfirmed(' Research ', 'Research'), false);
});
