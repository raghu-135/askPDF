import assert from 'node:assert/strict';
import test from 'node:test';

import {
  isRuntimeOperationEnabled,
  runtimeInterruptResponseOperation,
  runtimeOperationAvailability,
  isCurrentRuntimeCapabilityRequest,
  runtimeCapabilityResponseMatchesRun,
  TASK_CONTROL_CATALOG,
} from '../src/lib/runtime-capabilities.ts';

const response = (operations) => ({
  runtime_available: true,
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

test('missing operations are hidden', () => {
  const result = runtimeOperationAvailability(response({}), 'run.resume');

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

test('cancellation pending is preserved as a backend disabled reason', () => {
  const result = runtimeOperationAvailability(response({
    'run.cancel': {
      support: 'conditional',
      enabled: false,
      disabled_reason: 'cancellation_pending',
    },
  }), 'run.cancel');

  assert.equal(result.disabledReason, 'cancellation_pending');
});

test('capability discovery failure fails closed', () => {
  assert.equal(isRuntimeOperationEnabled(null, 'run.cancel'), false);
  assert.equal(runtimeOperationAvailability({ runtime_available: false, capabilities: null }, 'run.cancel').visible, false);
});

test('distinct runtime interactions are evaluated independently', () => {
  const capabilities = response({
    'run.cancel': { support: 'native', enabled: true },
    'task.pause': { support: 'conditional', enabled: true },
    'run.resume': { support: 'conditional', enabled: true },
    'task.retry': { support: 'conditional', enabled: true },
    'run.approval.respond': { support: 'native', enabled: true },
    'run.send_followup': { support: 'native', enabled: true },
    'run.interrupt_with_input': { support: 'emulated', enabled: true },
    'run.steer_live': { support: 'unsupported', enabled: false, disabled_reason: 'runtime_capability_unsupported' },
    'run.update_state': { support: 'conditional', enabled: false, disabled_reason: 'run_not_checkpoint_boundary' },
  });

  assert.equal(isRuntimeOperationEnabled(capabilities, 'run.cancel'), true);
  assert.equal(isRuntimeOperationEnabled(capabilities, 'task.pause'), true);
  assert.equal(isRuntimeOperationEnabled(capabilities, 'run.resume'), true);
  assert.equal(isRuntimeOperationEnabled(capabilities, 'task.retry'), true);
  assert.equal(isRuntimeOperationEnabled(capabilities, 'run.approval.respond'), true);
  assert.equal(isRuntimeOperationEnabled(capabilities, 'run.send_followup'), true);
  assert.equal(isRuntimeOperationEnabled(capabilities, 'run.interrupt_with_input'), true);
  assert.equal(isRuntimeOperationEnabled(capabilities, 'run.steer_live'), false);
  assert.equal(isRuntimeOperationEnabled(capabilities, 'run.update_state'), false);
});

test('interrupt response operations fail closed when missing or unknown', () => {
  assert.equal(runtimeInterruptResponseOperation({ response_operation: 'run.resume' }), 'run.resume');
  assert.equal(runtimeInterruptResponseOperation({ response_operation: 'run.approval.respond' }), 'run.approval.respond');
  assert.equal(runtimeInterruptResponseOperation({}), undefined);
  assert.equal(runtimeInterruptResponseOperation({ response_operation: 'run.unknown' }), undefined);
});

test('paused task resume uses task.resume rather than runtime run.resume', () => {
  const capabilities = response({
    'task.resume': { support: 'conditional', enabled: true },
  });
  const resumeControl = TASK_CONTROL_CATALOG.find((control) => control.action === 'resume');

  assert.equal(resumeControl?.operation, 'task.resume');
  assert.equal(runtimeOperationAvailability(capabilities, resumeControl.operation).enabled, true);
  assert.equal(runtimeOperationAvailability(capabilities, 'run.resume').visible, false);
});

test('capability responses are applied only to their selected run', () => {
  assert.equal(runtimeCapabilityResponseMatchesRun({ resource: 'run', run_id: 'run-2' }, 'run-1'), false);
  assert.equal(runtimeCapabilityResponseMatchesRun({ resource: 'run', run_id: 'run-1' }, 'run-1'), true);
  assert.equal(isCurrentRuntimeCapabilityRequest(3, 4), false);
  assert.equal(isCurrentRuntimeCapabilityRequest(4, 4), true);
});

test('runtime binding requirement is preserved as descriptor metadata', () => {
  const descriptor = {
    support: 'native',
    owner: 'runtime',
    enabled: true,
    requires_runtime_binding: true,
  };
  const result = runtimeOperationAvailability(response({ 'run.cancel': descriptor }), 'run.cancel');
  assert.equal(result.descriptor.requires_runtime_binding, true);
});
