import assert from 'node:assert/strict';
import test from 'node:test';

import { HOME_INSTRUCTION_SECTIONS } from '../src/lib/home-instructions.ts';

test('home instructions cover the major app workflows', () => {
  const titles = HOME_INSTRUCTION_SECTIONS.map((section) => section.title);
  assert.deepEqual(titles, [
    'Projects and Threads',
    'Documents and Browser Sources',
    'Chat and Retrieval',
    'Memory & Settings',
    'Agent Workflows',
    'Review, Trace, and Playback',
  ]);

  const text = HOME_INSTRUCTION_SECTIONS.flatMap((section) => section.items).join(' ');
  for (const phrase of [
    'Global memory',
    'Project memory',
    'Thread memory',
    'Agent Workflow Builder',
    'Debug Trace',
    'Browser tab',
    'model picker',
    'fork',
  ]) {
    assert.match(text, new RegExp(phrase));
  }
});
