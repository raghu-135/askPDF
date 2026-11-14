import test from 'node:test';
import assert from 'node:assert/strict';
import {
  toolApprovalDisplayName,
  toolApprovalSummaryLine,
  toolApprovalTitle,
} from '../src/lib/tool-approval-presentation.ts';

test('toolApprovalSummaryLine uses catalog display name and query', () => {
  const catalog = { search_web: { display_name: 'Internet Search', description: 'Live web search' } };
  assert.equal(
    toolApprovalSummaryLine('search_web', { query: 'USD EUR rate' }, catalog),
    'Search the web for: USD EUR rate',
  );
});

test('toolApprovalDisplayName falls back to humanized tool id', () => {
  assert.equal(toolApprovalDisplayName('publish_canvas', {}), 'Publish Canvas');
});

test('toolApprovalTitle uses catalog wording for web search', () => {
  const catalog = { search_web: { display_name: 'Internet Search', description: 'Live web search' } };
  assert.equal(toolApprovalTitle('search_web', catalog), 'Allow a web search?');
});