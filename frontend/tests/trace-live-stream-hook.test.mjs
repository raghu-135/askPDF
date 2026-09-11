import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { JSDOM } from 'jsdom';
import ts from 'typescript';

const source = readFileSync(new URL('../src/components/workbench/useTraceTabs.ts', import.meta.url), 'utf8');
const apiModule = 'data:text/javascript,' + encodeURIComponent([
  'export const streamAgentRunEvents = (...args) => globalThis.streamAgentRunEventsMock(...args);',
].join('\n'));
const configModule = 'data:text/javascript,' + encodeURIComponent('export const AGENT_SSE_RECONNECT_INTERVAL_MS = 1;');
const projectionModule = 'data:text/javascript,' + encodeURIComponent('export const buildLiveTraceView = (events) => ({ events });');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX },
}).outputText
  .replace("from 'react'", `from '${import.meta.resolve('react')}'`)
  .replace("from '../../lib/api'", `from '${apiModule}'`)
  .replace("from '../../lib/agent-ui-config'", `from '${configModule}'`)
  .replace("from '../../lib/trace-tabs'", `from '${new URL('../src/lib/trace-tabs.ts', import.meta.url)}'`)
  .replace("from '../agent-debug/agent-trace-projection'", `from '${projectionModule}'`);
const { default: useTraceTabs } = await import('data:text/javascript,' + encodeURIComponent(compiled));

test('live trace ownership survives source-panel removal and reconnects from its cursor', async () => {
  const dom = new JSDOM('<div id="root"></div>');
  globalThis.window = dom.window;
  globalThis.document = dom.window.document;
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const requests = [];
  globalThis.streamAgentRunEventsMock = (runId, threadId, afterSequence, onEvent, signal) => new Promise((resolve, reject) => {
    const request = { runId, threadId, afterSequence, onEvent, signal, resolve, reject };
    requests.push(request);
    signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true });
  });
  let current;
  function Harness({ showSource }) {
    current = useTraceTabs();
    return React.createElement('div', null, showSource ? React.createElement('span', null, 'source panel') : null);
  }
  const root = createRoot(document.getElementById('root'));
  const render = async (showSource) => act(async () => root.render(React.createElement(Harness, { showSource })));
  const flush = async () => act(async () => new Promise((resolve) => window.setTimeout(resolve, 140)));
  try {
    await render(true);
    await act(async () => current.openTrace({
      id: 'run-1',
      threadId: 'thread-1',
      messageId: 'task:run-1',
      label: 'Deep Research · attempt 1',
      status: 'running',
      running: true,
      liveEventSource: 'agent_run_events',
    }));
    assert.equal(requests.length, 1);
    assert.equal(requests[0].afterSequence, 0);

    await act(async () => requests[0].onEvent({
      id: 'event-1',
      event: 'operation.started',
      data: { sequence: 1, operation_id: 'planner', terminal: false },
    }));
    await flush();
    assert.equal(current.traceTabs[0].liveTraceView.events.length, 1);

    await render(false);
    assert.equal(requests[0].signal.aborted, false);
    assert.equal(current.traceTabs[0].liveTraceView.events.length, 1);

    await act(async () => requests[0].resolve());
    await act(async () => new Promise((resolve) => window.setTimeout(resolve, 5)));
    assert.equal(requests.length, 2);
    assert.equal(requests[1].afterSequence, 1);

    await act(async () => {
      requests[1].onEvent({
        id: 'event-2',
        event: 'run.completed',
        data: { sequence: 2, terminal: true },
      });
      requests[1].resolve();
    });
    assert.equal(current.traceTabs[0].running, false);
    assert.equal(current.traceTabs[0].status, 'completed');
    assert.equal(current.traceTabs[0].liveTraceView.events.length, 2);
    await act(async () => new Promise((resolve) => window.setTimeout(resolve, 5)));
    assert.equal(requests.length, 2);
  } finally {
    await act(async () => root.unmount());
    dom.window.close();
    delete globalThis.streamAgentRunEventsMock;
    delete globalThis.window;
    delete globalThis.document;
    delete globalThis.IS_REACT_ACT_ENVIRONMENT;
  }
});
