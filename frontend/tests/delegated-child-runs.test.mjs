import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  delegatedChildRunLinkLabel,
  delegatedChildRunsFromTechnical,
} from '../src/lib/delegated-child-runs.ts';

test('delegatedChildRunsFromTechnical prefers list field', () => {
  const rows = delegatedChildRunsFromTechnical({
    delegated_child_runs: [
      { run_id: 'a', workflow_id: 'router_rag_agent', order: 1 },
      { run_id: 'b', workflow_id: 'plan_execute_rag_agent', order: 2 },
    ],
  });
  assert.equal(rows.length, 2);
  assert.equal(rows[1].workflow_id, 'plan_execute_rag_agent');
});

test('delegatedChildRunLinkLabel formats workflow id', () => {
  assert.equal(delegatedChildRunLinkLabel('router_rag_agent'), 'Delegated: router_rag_agent');
});