import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildDocumentWorkspaceTabs,
  buildHomeWorkspaceTabs,
  buildProjectWorkspaceTabs,
  isBrowserWorkspaceActive,
  projectWorkspaceLandingTabId,
  selectedWorkspaceTabValue,
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

test('document workspace tabs include memory, browser, documents, canvas, embeddings, and debug trace', () => {
  const tabs = buildDocumentWorkspaceTabs({
    enabled: true,
    documents: [document],
    traces: [{ running: true }],
    includeResearchCanvas: true,
    canvasCount: 2,
  });

  assert.deepEqual(
    tabs.map((tab) => tab.kind),
    ['memory', 'browser', 'document', 'research_canvas', 'embeddings', 'trace'],
  );
  assert.equal(tabs[0].id, 'memory-tab');
  assert.equal(tabs[1].id, 'browser-tab');
  assert.equal(tabs[2].id, 'file-1');
  assert.equal(tabs[3].id, 'research-canvas-tab');
  assert.equal(tabs[3].count, 2);
  assert.equal(tabs[4].id, 'embeddings-tab');
  assert.equal(tabs[5].id, 'trace-tab');
  assert.equal(tabs[5].status, 'running');
  assert.equal(tabs[5].count, 1);
});

test('document workspace tabs are empty when disabled', () => {
  assert.deepEqual(buildDocumentWorkspaceTabs({ enabled: false, documents: [document], traces: [] }), []);
});

test('home workspace opens instructions first and memory explicitly', () => {
  const tabs = buildHomeWorkspaceTabs();
  assert.deepEqual(tabs.map((tab) => tab.kind), ['home', 'memory']);
  assert.deepEqual(tabs.map((tab) => tab.id), ['home-tab', 'memory-tab']);
});

test('project workspace opens overview first and keeps browser off the landing tab', () => {
  const tabs = buildProjectWorkspaceTabs([document]);
  assert.deepEqual(tabs.map((tab) => tab.kind), ['project', 'memory', 'browser', 'document']);
  assert.deepEqual(tabs.map((tab) => tab.id), ['project-tab', 'memory-tab', 'browser-tab', 'file-1']);
});

test('project landing stays on overview instead of leftover browser or a document', () => {
  assert.equal(projectWorkspaceLandingTabId([document]), 'project-tab');
  assert.equal(projectWorkspaceLandingTabId([]), 'project-tab');
});

test('browser workspace follows the Browser tab, not a leftover active flag', () => {
  assert.equal(isBrowserWorkspaceActive({ activeTabId: 'browser-tab', isBrowserActive: false }), true);
  assert.equal(isBrowserWorkspaceActive({ activeTabId: 'file-1', isBrowserActive: true }), false);
  assert.equal(isBrowserWorkspaceActive({ activeTabId: 'project-tab', isBrowserActive: true }), false);
  assert.equal(isBrowserWorkspaceActive({ activeTabId: 'file-1', isBrowserActive: false }), false);
});

test('selected workspace tab value ignores stale ids from a previous workspace', () => {
  assert.equal(selectedWorkspaceTabValue(
    [{ id: 'project-tab' }, { id: 'memory-tab' }, { id: 'browser-tab' }],
    'home-tab',
  ), false);
  assert.equal(selectedWorkspaceTabValue(
    [{ id: 'project-tab' }, { id: 'memory-tab' }, { id: 'browser-tab' }],
    'project-tab',
  ), 'project-tab');
});
