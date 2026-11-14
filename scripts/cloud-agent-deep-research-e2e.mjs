#!/usr/bin/env node
/**
 * Live Deep Research API flow: create task, start, poll, optional steer/retry at review.
 * Usage: node scripts/cloud-agent-deep-research-e2e.mjs [--hermes-retry]
 */
import { createRequire } from 'node:module';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.dirname(fileURLToPath(new URL('.', import.meta.url)));
const require = createRequire(path.join(repoRoot, 'frontend/package.json'));
const { chromium } = require('@playwright/test');
const envText = readFileSync(path.join(repoRoot, '.env'), 'utf8');
const admin = envText.match(/^ASKPDF_ADMIN_TOKEN=(.+)$/m)?.[1]?.trim();
if (!admin) throw new Error('ASKPDF_ADMIN_TOKEN missing in .env');

const ART = process.env.ASKPDF_E2E_ARTIFACTS || '/opt/cursor/artifacts';
mkdirSync(ART, { recursive: true });
const API = process.env.ASKPDF_E2E_API_URL || 'http://127.0.0.1:8000';
const BASE = process.env.ASKPDF_E2E_BASE_URL || 'http://127.0.0.1:3000';
const MODEL = process.env.ASKPDF_E2E_LLM_MODEL || 'openai/gpt-4.1-mini';
const PDF = process.env.ASKPDF_E2E_PDF || path.join(repoRoot, 'rag_service/tests/01030000000000.pdf');
const hermesRetry = process.argv.includes('--hermes-retry');

const auth = { Authorization: `Bearer ${admin}`, 'Content-Type': 'application/json' };
const started = Date.now();
const log = (msg) => process.stderr.write(`${msg}\n`);

async function json(method, url, body, headers = {}) {
  const response = await fetch(url, {
    method,
    headers: { ...auth, ...headers, ...(body ? {} : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await response.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { raw: text }; }
  if (!response.ok) throw new Error(`${method} ${url} ${response.status}: ${text.slice(0, 400)}`);
  return data;
}

async function uploadPdf(threadId) {
  const form = new FormData();
  const blob = new Blob([readFileSync(PDF)], { type: 'application/pdf' });
  form.append('file', blob, path.basename(PDF));
  const response = await fetch(`${API}/api/threads/${threadId}/files/upload`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${admin}` },
    body: form,
  });
  if (!response.ok) throw new Error(`upload failed ${response.status}`);
  return response.json();
}

async function pollTask(taskId, threadId, { timeoutMs = 900_000 } = {}) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const { task } = await json('GET', `${API}/api/agent-tasks/${taskId}?thread_id=${threadId}`);
    writeFileSync(path.join(ART, 'dr-live-task.json'), JSON.stringify({ task }, null, 2));
    if (['awaiting_approval', 'recovery_required', 'failed', 'completed', 'cancelled'].includes(task.status)) {
      return task;
    }
    await new Promise((r) => setTimeout(r, 8000));
  }
  throw new Error('poll timeout');
}

async function screenshot(label) {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  await page.goto(`${BASE}/`, { waitUntil: 'domcontentloaded', timeout: 120_000 });
  await page.waitForTimeout(2000);
  const out = path.join(ART, `${label}.png`);
  await page.screenshot({ path: out, fullPage: false });
  await browser.close();
  return out;
}

const project = (await json('POST', `${API}/api/projects`, {
  name: `dr-e2e-${Date.now()}`,
  description: 'cloud agent e2e',
  embedding_model: 'BAAI/bge-m3',
})).id;

const thread = (await json('POST', `${API}/api/projects/${project}/threads`, { name: 'dr-e2e' })).id;
await uploadPdf(thread);

const definitionId = hermesRetry ? 'hermes_rag_agent' : 'deep_research_agent';
const objective = hermesRetry
  ? 'Summarize the attached PDF in two sentences with a citation.'
  : 'List at least 12 distinct limitations of the methodology in the attached PDF, each with a citation.';

const idem = `dr-e2e-${Date.now()}`;
const created = await json('POST', `${API}/api/threads/${thread}/agent-tasks`, {
  definition_id: definitionId,
  objective,
  llm_model: MODEL,
}, { 'Idempotency-Key': idem });

const taskId = created.task.id;
await json('POST', `${API}/api/agent-tasks/${taskId}/commands/start?thread_id=${thread}`, { expected_version: created.task.version }, {
  'Idempotency-Key': `start-${taskId}`,
});

let task = await pollTask(taskId, thread);
const result = {
  task_id: taskId,
  thread_id: thread,
  definition_id: definitionId,
  status: task.status,
  terminal_reason: task.terminal_reason,
  phase: task.current_phase,
  elapsed_ms: Date.now() - started,
  screenshot: await screenshot(hermesRetry ? 'hermes-flow' : 'deep-research-flow'),
};

if (hermesRetry && task.status === 'failed') {
  const retried = await json('POST', `${API}/api/agent-tasks/${taskId}/commands/retry?thread_id=${thread}`, {
    expected_version: task.version,
  }, { 'Idempotency-Key': `retry-${taskId}-${Date.now()}` });
  task = await pollTask(taskId, thread, { timeoutMs: 600_000 });
  result.retry = { command_status: retried?.task?.status, final_status: task.status };
}

if (!hermesRetry && task.status === 'awaiting_approval') {
  const full = await json('GET', `${API}/api/agent-tasks/${taskId}?thread_id=${thread}`);
  task = full.task;
  const runId = task.active_run_id;
  const interrupt = task.active_run?.pending_interrupt;
  const gaps = interrupt?.gaps?.length ?? 0;
  const issues = (interrupt?.warnings || []).flatMap((w) => (w?.details?.issues) || []).length;
  result.review = { gaps, critic_issues: issues, interrupt_type: interrupt?.type };
  const steerIdem = `steer-${taskId}-${Date.now()}`;
  try {
    if (interrupt?.type === 'budget_review') {
      await json('POST', `${API}/api/agent-tasks/${taskId}/budget-review/responses?thread_id=${thread}`, {
        run_id: runId,
        interrupt_id: interrupt.interrupt_id,
        expected_version: task.version,
        decision: 'steer',
        guidance: 'Focus only on PDF pages 1–3 for the next tranche.',
      }, { 'Idempotency-Key': steerIdem });
    } else {
      await json('POST', `${API}/api/agent-tasks/${taskId}/result-review/responses?thread_id=${thread}`, {
        run_id: runId,
        interrupt_id: interrupt.interrupt_id,
        expected_version: task.version,
        decision: 'retry_with_input',
        followup_input: 'Focus only on PDF pages 1–3 for the next tranche.',
      }, { 'Idempotency-Key': steerIdem });
    }
    result.steer_or_retry = 'ok';
  } catch (err) {
    result.steer_or_retry = String(err.message).slice(0, 300);
  }
}

writeFileSync(path.join(ART, 'dr-e2e-result.json'), JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));