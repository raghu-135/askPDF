import assert from 'node:assert/strict';
import test from 'node:test';

import {
  isInterruptAlreadyResolvedError,
  shouldDismissRecoveredHumanReview,
} from '../src/lib/human-review-resume-recovery.ts';

test('isInterruptAlreadyResolvedError matches resume API wording', () => {
  assert.equal(
    isInterruptAlreadyResolvedError('This interrupt has already been resolved.'),
    true,
  );
  assert.equal(isInterruptAlreadyResolvedError('interrupt_already_resolved'), true);
  assert.equal(isInterruptAlreadyResolvedError('network error'), false);
});

test('shouldDismissRecoveredHumanReview when interrupt is gone or not pending', () => {
  assert.equal(shouldDismissRecoveredHumanReview(null), true);
  assert.equal(shouldDismissRecoveredHumanReview({ pending_interrupt: null }), true);
  assert.equal(
    shouldDismissRecoveredHumanReview({
      pending_interrupt: { interrupt_id: 'i-1', status: 'approved' },
    }),
    true,
  );
  assert.equal(
    shouldDismissRecoveredHumanReview({
      pending_interrupt: { interrupt_id: 'i-1', status: 'pending' },
    }),
    false,
  );
});