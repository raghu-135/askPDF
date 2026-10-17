import assert from 'node:assert/strict';
import test from 'node:test';

import {
  citationHighlightIds,
  shouldClearPdfHighlightOnDocumentActivate,
  shouldPreserveCitationHighlight,
} from '../src/lib/document-citation-navigation.ts';
import { shouldClearCurrentOnSourceKeyChange } from '../src/lib/player-source-key.ts';

test('citationHighlightIds prefers sentenceIds array', () => {
  assert.deepEqual(
    citationHighlightIds({ fileHash: 'a'.repeat(32), sentenceIds: [226, 227, 228], sentenceId: 217 }),
    [226, 227, 228],
  );
});

test('shouldPreserveCitationHighlight matches pending citation file hash to document tab', () => {
  const fileHash = 'f'.repeat(32);
  const pending = { fileHash, sentenceIds: [1] };
  assert.equal(shouldPreserveCitationHighlight(pending, { fileHash }), true);
  assert.equal(shouldPreserveCitationHighlight(pending, { fileHash: 'b'.repeat(32) }), false);
  assert.equal(shouldPreserveCitationHighlight(null, { fileHash }), false);
});

test('canvas citation open from canvas tab keeps pending target for documents activation', () => {
  const fileHash = 'c'.repeat(32);
  const documentTab = { id: 'doc-tab-1', fileHash };
  const target = { fileHash, sentenceIds: [226, 227, 228], play: false };
  let currentPdfId = null;
  let highlightIds = null;
  let pending = target;

  const applyHighlight = (value) => {
    const ids = citationHighlightIds(value);
    highlightIds = ids;
    currentPdfId = ids[0] ?? null;
  };

  const handleActiveDocumentChange = (documentId, tabs) => {
    const tab = tabs.find((item) => item.id === documentId);
    if (shouldPreserveCitationHighlight(pending, tab)) {
      applyHighlight(pending);
    } else {
      pending = null;
      currentPdfId = null;
      highlightIds = null;
    }
  };

  const handleOpenDocumentCitation = (citationTarget, tabs) => {
    const tab = tabs.find((item) => item.fileHash === citationTarget.fileHash);
    if (!tab) return;
    pending = citationTarget;
    handleActiveDocumentChange(tab.id, tabs);
  };

  handleOpenDocumentCitation(target, [documentTab]);
  assert.equal(currentPdfId, 226);
  assert.deepEqual(highlightIds, [226, 227, 228]);
  assert.equal(pending, target);
});

test('nested document tab re-activation does not clear an applied citation highlight', () => {
  const fileHash = 'e'.repeat(32);
  const documentTab = { id: 'doc-tab-1', fileHash };
  const target = { fileHash, sentenceIds: [227], play: false };
  let currentPdfId = 227;
  let pending = null;

  const handleActiveDocumentChange = (documentId, previousDocumentId, tabs) => {
    const tab = tabs.find((item) => item.id === documentId);
    if (shouldPreserveCitationHighlight(pending, tab)) {
      currentPdfId = citationHighlightIds(pending)[0];
      return;
    }
    if (shouldClearPdfHighlightOnDocumentActivate(previousDocumentId, documentId, pending, tab)) {
      currentPdfId = null;
    }
  };

  handleActiveDocumentChange(documentTab.id, documentTab.id, [documentTab]);
  assert.equal(currentPdfId, 227);
});

test('canvas citation survives documents player mount without clearing current id', () => {
  const fileHash = 'd'.repeat(32);
  const target = { fileHash, sentenceIds: [226], play: false };
  let currentPdfId = citationHighlightIds(target)[0];
  let previousSourceKey = null;
  const sourceKey = `pdf:${fileHash}`;

  const onCurrentChange = (id) => {
    currentPdfId = id;
  };

  if (shouldClearCurrentOnSourceKeyChange(previousSourceKey, sourceKey)) {
    onCurrentChange(null);
  }
  previousSourceKey = sourceKey;

  assert.equal(currentPdfId, 226);
});
