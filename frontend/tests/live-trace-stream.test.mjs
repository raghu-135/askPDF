import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

const source = readFileSync(new URL('../src/lib/live-trace-stream.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
  .replace("from './trace-tabs'", `from '${new URL('../src/lib/trace-tabs.ts', import.meta.url)}'`)
  .replace("from './agent-execution-stream'", `from '${new URL('../src/lib/agent-execution-stream.ts', import.meta.url)}'`);
const { LiveTraceStreamController, liveTraceRunIdFromEvent } = await import('data:text/javascript,' + encodeURIComponent(compiled));

test('live stream adopts a run id from event data and ignores placeholders', () => {
  const stream = new LiveTraceStreamController('temp-assistant-1');
  assert.equal(stream.snapshot().runId, '');
  assert.equal(liveTraceRunIdFromEvent({
    id: 1,
    event: 'operation.started',
    data: { operation_id: 'planner' },
  }), undefined);

  const snapshot = stream.append({
    id: 1,
    event: 'operation.started',
    data: { operation_id: 'planner', run_id: 'run-abc' },
  });
  assert.equal(snapshot.runId, 'run-abc');
  assert.equal(snapshot.events.length, 1);
});
