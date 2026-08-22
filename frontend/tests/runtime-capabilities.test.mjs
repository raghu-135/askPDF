import assert from 'node:assert/strict';
import test from 'node:test';

import {
  isRuntimeOperationEnabled,
  runtimeOperationAvailability,
} from '../src/lib/runtime-capabilities.ts';

const response = (operations) => ({
  available: true,
  capabilities: { operations },
});

test('unsupported operations are hidden and never enabled', () => {
  const result = runtimeOperationAvailability(response({
    'run.steer_live': {
      support: 'unsupported',
      enabled: false,
      disabled_reason: 'runtime_capability_unsupported',
    },
  }), 'run.steer_live');

  assert.deepEqual(result, { visible: false, enabled: false });
});

test('conditional operations remain visible with their backend reason', () => {
  const result = runtimeOperationAvailability(response({
    'run.cancel': {
      support: 'conditional',
      enabled: false,
      disabled_reason: 'run_terminal',
    },
  }), 'run.cancel');

  assert.equal(result.visible, true);
  assert.equal(result.enabled, false);
  assert.equal(result.disabledReason, 'run_terminal');
});

test('capability discovery failure fails closed', () => {
  assert.equal(isRuntimeOperationEnabled(null, 'run.cancel'), false);
  assert.equal(runtimeOperationAvailability({ available: false, capabilities: null }, 'run.cancel').visible, false);
});

test('distinct runtime interactions are evaluated independently', () => {
  const capabilities = response({
    'run.send_followup': { support: 'native', enabled: true },
    'run.interrupt_with_input': { support: 'emulated', enabled: true },
    'run.steer_live': { support: 'unsupported', enabled: false, disabled_reason: 'runtime_capability_unsupported' },
    'run.update_state': { support: 'conditional', enabled: false, disabled_reason: 'run_not_checkpoint_boundary' },
  });

  assert.equal(isRuntimeOperationEnabled(capabilities, 'run.send_followup'), true);
  assert.equal(isRuntimeOperationEnabled(capabilities, 'run.interrupt_with_input'), true);
  assert.equal(isRuntimeOperationEnabled(capabilities, 'run.steer_live'), false);
  assert.equal(isRuntimeOperationEnabled(capabilities, 'run.update_state'), false);
});
