import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildDocumentWorkspaceTabs,
  buildHomeWorkspaceTabs,
  buildProjectWorkspaceTabs,
  isBrowserWorkspaceActive,
  traceWorkspaceStatus,
} from '../src/lib/document-tabs.ts';

const document = {
  id: 'file-1',
  fileName: 'Paper.pdf',
  fileHash: 'file-1',
  downloadUrl: '/download/file-1',
  sentences: null,
};

test('trace workspace status prefers failed over running over idle', () => {
  assert.equal(traceWorkspaceStatus([]), 'idle');
  assert.equal(traceWorkspaceStatus([{ running: true }]), 'running');
  assert.equal(traceWorkspaceStatus([{ running: true }, { error: 'boom' }]), 'failed');
});

test('document workspace tabs include memory, browser, documents, and debug trace', () => {
  const tabs = buildDocumentWorkspaceTabs({
    enabled: true,
    documents: [document],
    traces: [{ running: true }],
  });

  assert.deepEqual(tabs.map((tab) => tab.kind), ['memory', 'browser', 'document', 'trace']);
  assert.equal(tabs[0].id, 'memory-tab');
  assert.equal(tabs[1].id, 'browser-tab');
  assert.equal(tabs[2].id, 'file-1');
  assert.equal(tabs[3].id, 'trace-tab');
  assert.equal(tabs[3].status, 'running');
  assert.equal(tabs[3].count, 1);
});

test('document workspace tabs are empty when disabled', () => {
  assert.deepEqual(buildDocumentWorkspaceTabs({ enabled: false, documents: [document], traces: [] }), []);
});

test('home workspace opens memory', () => {
  const tabs = buildHomeWorkspaceTabs();
  assert.deepEqual(tabs.map((tab) => tab.kind), ['memory']);
  assert.deepEqual(tabs.map((tab) => tab.id), ['memory-tab']);
});

test('project workspace includes memory, browser, and shared documents without debug trace', () => {
  const tabs = buildProjectWorkspaceTabs([document]);
  assert.deepEqual(tabs.map((tab) => tab.kind), ['memory', 'browser', 'document']);
  assert.deepEqual(tabs.map((tab) => tab.id), ['memory-tab', 'browser-tab', 'file-1']);
});

test('browser workspace is inactive after switching to a PDF tab', () => {
  assert.equal(isBrowserWorkspaceActive({ activeTabId: 'browser-tab', isBrowserActive: false }), true);
  assert.equal(isBrowserWorkspaceActive({ activeTabId: 'file-1', isBrowserActive: true }), true);
  assert.equal(isBrowserWorkspaceActive({ activeTabId: 'file-1', isBrowserActive: false }), false);
});
