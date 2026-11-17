import test from 'node:test';
import assert from 'node:assert/strict';
import {
  isAgentRunResumeProductTerminal,
  parseAgentRunResumeStreamEvent,
} from '../src/lib/agent-run-resume-stream.ts';

test('parseAgentRunResumeStreamEvent normalizes __result__ payloads', () => {
  const parsed = parseAgentRunResumeStreamEvent('__result__', {
    response: {
      agent_run: { status: 'completed', result_json: { answer: 'done' } },
      outcome: {},
    },
  });
  assert.equal(parsed?.response.answer, 'done');
});

test('parseAgentRunResumeStreamEvent reads top-level agent_run on run.completed', () => {
  const parsed = parseAgentRunResumeStreamEvent('run.completed', {
    agent_run: { status: 'completed', result_json: { final_answer: 'final' } },
    outcome: {},
  });
  assert.equal(parsed?.response.answer, 'final');
});

test('parseAgentRunResumeStreamEvent normalizes nested response envelopes', () => {
  const parsed = parseAgentRunResumeStreamEvent('run.completed', {
    response: {
      agent_run: { status: 'completed' },
      outcome: { answer: 'from outcome' },
    },
  });
  assert.equal(parsed?.response.answer, 'from outcome');
});

test('isAgentRunResumeProductTerminal ignores non-terminal run.completed agent_run', () => {
  assert.equal(
    isAgentRunResumeProductTerminal('run.completed', {
      agent_run: { status: 'running' },
      outcome: {},
    }),
    false,
  );
});

test('isAgentRunResumeProductTerminal ignores raw runtime run.completed', () => {
  assert.equal(
    isAgentRunResumeProductTerminal('run.completed', { final_output: { answer: 'x' } }),
    false,
  );
  assert.equal(
    isAgentRunResumeProductTerminal('run.completed', {
      response: { agent_run: { status: 'completed' }, outcome: {} },
    }),
    true,
  );
});

test('isAgentRunResumeProductTerminal ignores human-review boundary without agent_run', () => {
  assert.equal(
    isAgentRunResumeProductTerminal('interrupt.requested', {
      run_id: 'run-1',
      status: 'awaiting_human',
      response: { status: 'awaiting_human', pending_interrupt: { interrupt_id: 'i-1' } },
    }),
    false,
  );
  assert.equal(
    isAgentRunResumeProductTerminal('interrupt.requested', {
      agent_run: { status: 'awaiting_human', pending_interrupt: { interrupt_id: 'i-1' } },
    }),
    true,
  );
});

test('parseAgentRunResumeStreamEvent derives agent_run from pending_interrupt in boundary payload', () => {
  const parsed = parseAgentRunResumeStreamEvent('interrupt.requested', {
    response: {
      status: 'awaiting_human',
      pending_interrupt: { interrupt_id: 'i-1', type: 'tool_approval' },
    },
  });
  assert.equal(parsed?.response.agent_run?.status, 'awaiting_human');
  assert.equal(parsed?.response.agent_run?.pending_interrupt?.interrupt_id, 'i-1');
});