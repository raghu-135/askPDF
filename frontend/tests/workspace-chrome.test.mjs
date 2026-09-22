import assert from 'node:assert/strict';
import test from 'node:test';

import {
  WORKSPACE_CHROME_SEPARATOR,
  flattenWorkspaceChromeEntries,
  isWorkspaceChromeSeparator,
  splitWorkspaceChromeRows,
} from '../src/lib/workspace-chrome.ts';

test('workspace chrome separator is detected', () => {
  assert.equal(isWorkspaceChromeSeparator(WORKSPACE_CHROME_SEPARATOR), true);
  assert.equal(isWorkspaceChromeSeparator('search'), false);
});

test('workspace chrome rows split on separators and drop empty rows', () => {
  const rows = splitWorkspaceChromeRows([
    'search',
    WORKSPACE_CHROME_SEPARATOR,
    'annotate',
    'undo',
    WORKSPACE_CHROME_SEPARATOR,
    WORKSPACE_CHROME_SEPARATOR,
    'extras',
  ]);

  assert.deepEqual(rows, [
    ['search'],
    ['annotate', 'undo'],
    ['extras'],
  ]);
});

test('workspace chrome flatten skips separators and empty entries', () => {
  const flat = flattenWorkspaceChromeEntries([
    'search',
    null,
    WORKSPACE_CHROME_SEPARATOR,
    false,
    'play',
  ]);

  assert.deepEqual(flat, ['search', 'play']);
});
