import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { getChatComposerState } from '../src/lib/chat-composer-state.ts';
import {
  chatHumanReviewUsesStackedDecisionLayout,
  chatToolApprovalReplacesComposer,
  composerHumanReviewPending,
  shouldDismissToolApprovalOnAction,
} from '../src/lib/chat-tool-approval-ui.ts';

const readyInput = {
  loading: false,
  llmModel: 'gpt-test',
  isLlmModelValid: true,
  isLlmToolsSupported: true,
  isEmbeddingModelValid: null,
  indexingStatus: 'ready',
  hasInput: false,
};

const toolApprovalInterrupt = { type: 'tool_approval', interrupt_id: 'i-1' };
const finalAnswerInterrupt = { type: 'final_answer_review', interrupt_id: 'i-2' };

test('tool approval replaces composer instead of using stacked decision column', () => {
  assert.equal(chatToolApprovalReplacesComposer(toolApprovalInterrupt), true);
  assert.equal(chatHumanReviewUsesStackedDecisionLayout(toolApprovalInterrupt), false);
  assert.equal(chatToolApprovalReplacesComposer(finalAnswerInterrupt), false);
  assert.equal(chatHumanReviewUsesStackedDecisionLayout(finalAnswerInterrupt), true);
});

test('only non-tool reviews lock the chat composer', () => {
  assert.equal(composerHumanReviewPending(toolApprovalInterrupt), false);
  assert.equal(composerHumanReviewPending(finalAnswerInterrupt), true);
  assert.equal(composerHumanReviewPending(null), false);
});

test('composer re-enables immediately once pending tool approval is cleared', () => {
  assert.equal(shouldDismissToolApprovalOnAction(toolApprovalInterrupt), true);
  assert.equal(composerHumanReviewPending(null), false);
  const state = getChatComposerState({
    ...readyInput,
    humanReviewPending: false,
    loading: false,
    isEmbeddingModelValid: null,
  });
  assert.equal(state.status, 'embedding_checking');
  const ready = getChatComposerState({
    ...readyInput,
    humanReviewPending: false,
    loading: false,
    isEmbeddingModelValid: true,
  });
  assert.equal(ready.status, 'ready');
  assert.equal(ready.disabled, false);
});

test('non-tool review takes precedence over embedding checks in composer state', () => {
  const state = getChatComposerState({
    ...readyInput,
    humanReviewPending: composerHumanReviewPending(finalAnswerInterrupt),
    isEmbeddingModelValid: null,
    loading: true,
  });
  assert.equal(state.status, 'human_review_pending');
  assert.equal(state.placeholder, 'Approve or skip to continue');
});

test('bottom sheet scroll region is flex-constrained with a pinned footer sibling', () => {
  const source = readFileSync(
    new URL('../src/components/conversation/ResizableDecisionPanel.tsx', import.meta.url),
    'utf8',
  );
  const bottomSheetSection = source.slice(source.indexOf('if (bottomSheet)'));
  assert.match(bottomSheetSection, /data-testid="review-decision-scroll"/);
  assert.match(bottomSheetSection, /flex:\s*'1 1 0'/);
  assert.match(bottomSheetSection, /minHeight:\s*0/);
  assert.match(bottomSheetSection, /overflowY:\s*'auto'/);
  const footerIndex = bottomSheetSection.indexOf('{footerRow}');
  const scrollIndex = bottomSheetSection.indexOf('review-decision-scroll');
  assert.ok(scrollIndex < footerIndex, 'footer follows the scroll body');
  assert.match(bottomSheetSection, /review-decision-footer/);
});