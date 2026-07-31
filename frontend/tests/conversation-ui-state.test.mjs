import assert from 'node:assert/strict';
import test from 'node:test';

import {
  clampDecisionPanelRatio,
  getConversationComposerButtonState,
} from '../src/lib/conversation-ui-state.ts';

test('decision panel ratio remains within canonical bounds', () => {
  assert.equal(clampDecisionPanelRatio(-1), 0.16);
  assert.equal(clampDecisionPanelRatio(0.3), 0.3);
  assert.equal(clampDecisionPanelRatio(2), 0.58);
});

test('composer button preserves send spinner and stop transitions', () => {
  assert.deepEqual(
    getConversationComposerButtonState({
      disabled: true,
      busy: true,
      showStop: false,
      canStop: false,
      stopping: false,
      hasDraft: true,
      disableWhenEmpty: false,
    }),
    { mode: 'send', disabled: true, spinning: true },
  );
  assert.deepEqual(
    getConversationComposerButtonState({
      disabled: true,
      busy: true,
      showStop: true,
      canStop: false,
      stopping: false,
      hasDraft: true,
      disableWhenEmpty: false,
    }),
    { mode: 'stop', disabled: true, spinning: false },
  );
  assert.deepEqual(
    getConversationComposerButtonState({
      disabled: true,
      busy: true,
      showStop: true,
      canStop: true,
      stopping: true,
      hasDraft: true,
      disableWhenEmpty: false,
    }),
    { mode: 'stop', disabled: true, spinning: true },
  );
});

test('empty-draft policy is configurable per conversation controller', () => {
  const input = {
    disabled: false,
    busy: false,
    showStop: false,
    canStop: false,
    stopping: false,
    hasDraft: false,
  };
  assert.equal(getConversationComposerButtonState({ ...input, disableWhenEmpty: false }).disabled, false);
  assert.equal(getConversationComposerButtonState({ ...input, disableWhenEmpty: true }).disabled, true);
});
