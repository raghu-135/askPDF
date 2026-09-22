import assert from 'node:assert/strict';
import test from 'node:test';

import {
  defaultOverviewTabId,
  readWorkspaceResume,
  resolveWorkspaceResume,
  writeWorkspaceResume,
  WORKSPACE_RESUME_STORAGE_KEY,
} from '../src/lib/workspace-resume-state.ts';
import {
  DOCUMENTS_TAB_ID,
  PROJECT_OVERVIEW_TAB_ID,
  THREAD_OVERVIEW_TAB_ID,
} from '../src/lib/document-tabs.ts';

const memory = new Map();

const storage = {
  getItem: (key) => memory.get(key) ?? null,
  setItem: (key, value) => {
    memory.set(key, value);
  },
};

test('default overview tab id follows workspace context', () => {
  assert.equal(defaultOverviewTabId('thread:abc'), THREAD_OVERVIEW_TAB_ID);
  assert.equal(defaultOverviewTabId('project:abc'), PROJECT_OVERVIEW_TAB_ID);
  assert.equal(defaultOverviewTabId('home'), 'home-tab');
});

test('missing cache resolves to overview tab', () => {
  const resolved = resolveWorkspaceResume({
    contextKey: 'thread:abc',
    state: null,
    availableTabs: [{ id: THREAD_OVERVIEW_TAB_ID }, { id: DOCUMENTS_TAB_ID }],
    pdfTabs: [{ id: 'file-1' }],
  });
  assert.deepEqual(resolved, {
    tabId: THREAD_OVERVIEW_TAB_ID,
    documentId: null,
    traceId: null,
    canvasId: null,
  });
});

test('valid cached tab is restored', () => {
  const resolved = resolveWorkspaceResume({
    contextKey: 'thread:abc',
    state: {
      tabId: 'browser-tab',
      documentId: 'file-1',
      traceId: 'trace-1',
      canvasId: 'canvas-1',
    },
    availableTabs: [{ id: THREAD_OVERVIEW_TAB_ID }, { id: 'browser-tab' }],
    pdfTabs: [{ id: 'file-1' }],
    traceIds: ['trace-1'],
    canvasIds: ['canvas-1'],
  });
  assert.equal(resolved.tabId, 'browser-tab');
  assert.equal(resolved.documentId, null);
  assert.equal(resolved.traceId, null);
  assert.equal(resolved.canvasId, null);
});

test('documents tab falls back to first document when cached document is stale', () => {
  const resolved = resolveWorkspaceResume({
    contextKey: 'thread:abc',
    state: {
      tabId: DOCUMENTS_TAB_ID,
      documentId: 'missing-file',
    },
    availableTabs: [{ id: THREAD_OVERVIEW_TAB_ID }, { id: DOCUMENTS_TAB_ID }],
    pdfTabs: [{ id: 'file-2' }, { id: 'file-3' }],
  });
  assert.equal(resolved.tabId, DOCUMENTS_TAB_ID);
  assert.equal(resolved.documentId, 'file-2');
});

test('documents tab without files falls back to overview', () => {
  const resolved = resolveWorkspaceResume({
    contextKey: 'thread:abc',
    state: { tabId: DOCUMENTS_TAB_ID, documentId: 'file-1' },
    availableTabs: [{ id: THREAD_OVERVIEW_TAB_ID }, { id: DOCUMENTS_TAB_ID }],
    pdfTabs: [],
  });
  assert.equal(resolved.tabId, THREAD_OVERVIEW_TAB_ID);
});

test('memory tab is never persisted', () => {
  memory.clear();
  globalThis.window = { localStorage: storage };
  writeWorkspaceResume('thread:abc', {
    tabId: 'memory-tab',
    documentId: null,
    traceId: null,
    canvasId: null,
  });
  assert.equal(memory.has(WORKSPACE_RESUME_STORAGE_KEY), false);
});

test('read and write workspace resume round trip', () => {
  memory.clear();
  globalThis.window = { localStorage: storage };

  writeWorkspaceResume('project:42', {
    tabId: DOCUMENTS_TAB_ID,
    documentId: 'file-9',
    traceId: null,
    canvasId: null,
  });
  assert.deepEqual(readWorkspaceResume('project:42'), {
    tabId: DOCUMENTS_TAB_ID,
    documentId: 'file-9',
    traceId: null,
    canvasId: null,
  });

});
