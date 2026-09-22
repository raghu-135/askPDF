import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildDocumentWorkspaceTabs,
  buildHomeWorkspaceTabs,
  buildProjectWorkspaceTabs,
  DOCUMENTS_TAB_ID,
  isBrowserWorkspaceActive,
  isDocumentsWorkspaceActive,
  projectWorkspaceLandingTabId,
  selectedWorkspaceTabValue,
  traceWorkspaceStatus,
} from '../src/lib/document-tabs.ts';

test('trace workspace status prefers failed over running over idle', () => {
  assert.equal(traceWorkspaceStatus([]), 'idle');
  assert.equal(traceWorkspaceStatus([{ running: true }]), 'running');
  assert.equal(traceWorkspaceStatus([{ running: true }, { error: 'boom' }]), 'failed');
});

test('document workspace tabs include memory, documents, browser, canvas, embeddings, and debug trace', () => {
  const tabs = buildDocumentWorkspaceTabs({
    enabled: true,
    documentCount: 3,
    traces: [{ running: true }],
    includeResearchCanvas: true,
    canvasCount: 2,
  });

  assert.deepEqual(
    tabs.map((tab) => tab.kind),
    ['memory', 'documents', 'browser', 'research_canvas', 'embeddings', 'trace'],
  );
  assert.equal(tabs[0].id, 'memory-tab');
  assert.equal(tabs[1].id, DOCUMENTS_TAB_ID);
  assert.equal(tabs[1].count, 3);
  assert.equal(tabs[2].id, 'browser-tab');
  assert.equal(tabs[3].id, 'research-canvas-tab');
  assert.equal(tabs[3].count, 2);
  assert.equal(tabs[4].id, 'embeddings-tab');
  assert.equal(tabs[5].id, 'trace-tab');
  assert.equal(tabs[5].status, 'running');
  assert.equal(tabs[5].count, 1);
});

test('document workspace tabs are empty when disabled', () => {
  assert.deepEqual(buildDocumentWorkspaceTabs({ enabled: false, documentCount: 1, traces: [] }), []);
});

test('home workspace opens instructions first and memory explicitly', () => {
  const tabs = buildHomeWorkspaceTabs();
  assert.deepEqual(tabs.map((tab) => tab.kind), ['home', 'memory']);
  assert.deepEqual(tabs.map((tab) => tab.id), ['home-tab', 'memory-tab']);
});

test('project workspace opens overview first and groups documents under documents tab', () => {
  const tabs = buildProjectWorkspaceTabs(2);
  assert.deepEqual(tabs.map((tab) => tab.kind), ['project', 'memory', 'documents', 'browser']);
  assert.deepEqual(tabs.map((tab) => tab.id), ['project-tab', 'memory-tab', DOCUMENTS_TAB_ID, 'browser-tab']);
  assert.equal(tabs[2].count, 2);
});

test('project landing stays on overview instead of leftover browser or a document', () => {
  assert.equal(projectWorkspaceLandingTabId([{ id: 'file-1' }]), 'project-tab');
  assert.equal(projectWorkspaceLandingTabId([]), 'project-tab');
});

test('browser workspace follows the Browser tab, not a leftover active flag', () => {
  assert.equal(isBrowserWorkspaceActive({ activeTabId: 'browser-tab', isBrowserActive: false }), true);
  assert.equal(isBrowserWorkspaceActive({ activeTabId: 'file-1', isBrowserActive: true }), false);
  assert.equal(isBrowserWorkspaceActive({ activeTabId: 'project-tab', isBrowserActive: true }), false);
  assert.equal(isBrowserWorkspaceActive({ activeTabId: 'file-1', isBrowserActive: false }), false);
});

test('documents workspace is active only on documents tab', () => {
  assert.equal(isDocumentsWorkspaceActive(DOCUMENTS_TAB_ID), true);
  assert.equal(isDocumentsWorkspaceActive('browser-tab'), false);
  assert.equal(isDocumentsWorkspaceActive('file-1'), false);
});

test('selected workspace tab value ignores stale ids from a previous workspace', () => {
  assert.equal(selectedWorkspaceTabValue(
    [{ id: 'project-tab' }, { id: 'memory-tab' }, { id: DOCUMENTS_TAB_ID }],
    'home-tab',
  ), false);
  assert.equal(selectedWorkspaceTabValue(
    [{ id: 'project-tab' }, { id: 'memory-tab' }, { id: DOCUMENTS_TAB_ID }],
    'project-tab',
  ), 'project-tab');
});
