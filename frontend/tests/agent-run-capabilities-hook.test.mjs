import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { JSDOM } from 'jsdom';
import ts from 'typescript';
import { isRuntimeOperationEnabled } from '../src/lib/runtime-capabilities.ts';

// Load the actual hook with only its network boundary replaced.
const source = readFileSync(new URL('../src/lib/use-agent-run-capabilities.ts', import.meta.url), 'utf8');
const networkModule = 'data:text/javascript,' + encodeURIComponent('export const getAgentRunCapabilities = (...args) => globalThis.capabilityRequest(...args);');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
  .replace("from 'react'", `from '${import.meta.resolve('react')}'`)
  .replace("from './api'", `from '${networkModule}'`)
  .replace("from './runtime-capabilities'", `from '${new URL('../src/lib/runtime-capabilities.ts', import.meta.url)}'`);
const { useAgentRunCapabilities } = await import('data:text/javascript,' + encodeURIComponent(compiled));

test('actual hook hides obsolete controls and handles failures, malformed data, and recovery', async () => {
  const dom = new JSDOM('<div id="root"></div>');
  globalThis.window = dom.window;
  globalThis.document = dom.window.document;
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const requests = [];
  globalThis.capabilityRequest = (run, thread) => new Promise((resolve, reject) => requests.push({ run, thread, resolve, reject }));
  let current;
  function Controls({ run, thread }) {
    current = useAgentRunCapabilities(run, thread);
    return isRuntimeOperationEnabled(current.capabilities, 'run.cancel') ? React.createElement('button', null, 'Cancel') : null;
  }
  const root = createRoot(document.getElementById('root'));
  const render = async (run, thread = 'thread-1') => act(async () => root.render(React.createElement(Controls, { run, thread })));
  const capabilities = run => ({ resource: 'run', run_id: run, runtime_available: true, capabilities: { operations: { 'run.cancel': { support: 'native', owner: 'runtime', enabled: true } } } });
  const visible = () => document.querySelector('button') !== null;
  try {
    await render('one');
    await act(async () => requests[0].resolve(capabilities('one')));
    assert.equal(visible(), true);
    await render('two');
    assert.equal(visible(), false);
    await render('three');
    await act(async () => requests[1].resolve(capabilities('two')));
    assert.equal(current.capabilities, null);
    await act(async () => requests[2].reject(new Error('offline')));
    assert.match(current.error, /offline/);
    assert.equal(visible(), false);
    let refresh;
    await act(async () => { refresh = current.refresh(); });
    await act(async () => { requests[3].resolve(capabilities('three')); await refresh; });
    assert.equal(visible(), true);
    await render('three', 'thread-2');
    assert.equal(visible(), false);
    await act(async () => requests[4].resolve({ ...capabilities('three'), runtime_available: 'true' }));
    assert.equal(current.capabilities, null);
    assert.equal(visible(), false);
    await render(null);
    assert.equal(current.capabilities, null);
  } finally {
    await act(async () => root.unmount());
    dom.window.close();
    delete globalThis.capabilityRequest;
    delete globalThis.window;
    delete globalThis.document;
    delete globalThis.IS_REACT_ACT_ENVIRONMENT;
  }
});
