import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildCanvasWorkspaceDeepLink,
  parseCanvasWorkspaceDeepLink,
  canvasDeepLinkResumeState,
  stripCanvasWorkspaceDeepLink,
} from '../src/lib/workspace-deep-link.ts';
import { RESEARCH_CANVAS_TAB_ID } from '../src/lib/canvas-spec.ts';

test('buildCanvasWorkspaceDeepLink encodes thread, project, and canvas', () => {
  const url = buildCanvasWorkspaceDeepLink({
    threadId: 'thread-abc',
    projectId: 'proj-xyz',
    canvasId: 'canvas-123',
  });
  assert.match(url, /\?thread=thread-abc/);
  assert.match(url, /project=proj-xyz/);
  assert.match(url, /canvas=canvas-123/);
});

test('parseCanvasWorkspaceDeepLink reads canvas links', () => {
  const parsed = parseCanvasWorkspaceDeepLink(
    '?thread=t1&project=p1&canvas=c9',
  );
  assert.deepEqual(parsed, {
    threadId: 't1',
    projectId: 'p1',
    canvasId: 'c9',
  });
});

test('parseCanvasWorkspaceDeepLink requires thread and canvas', () => {
  assert.equal(parseCanvasWorkspaceDeepLink('?canvas=c9'), null);
  assert.equal(parseCanvasWorkspaceDeepLink('?thread=t1'), null);
  assert.deepEqual(parseCanvasWorkspaceDeepLink('?thread=t1&canvas=c9'), {
    threadId: 't1',
    projectId: null,
    canvasId: 'c9',
  });
});

test('canvasDeepLinkResumeState selects research canvas tab', () => {
  const state = canvasDeepLinkResumeState('my-canvas');
  assert.equal(state.tabId, RESEARCH_CANVAS_TAB_ID);
  assert.equal(state.canvasId, 'my-canvas');
});

test('stripCanvasWorkspaceDeepLink is safe without a browser window', () => {
  assert.doesNotThrow(() => stripCanvasWorkspaceDeepLink());
});