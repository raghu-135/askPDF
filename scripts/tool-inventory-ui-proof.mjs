#!/usr/bin/env node
/**
 * Real-UI proof: tool catalog settings, 128k + 8k inventory chats, Debug Trace prompt tokens.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import {
  DEFAULT_API_URL,
  DEFAULT_BASE_URL,
  REAL_UI_VIEWPORT,
  ScreenshotGuard,
  apiJson,
  bootstrapProjectThreadViaApi,
  chatRateLimitPause,
  loadEnvAdmin,
  selectLlmModel,
  uploadPdfThroughUi,
  waitForFileReady,
  waitForComposerReady,
} from './cloud-agent-e2e-lib.mjs';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const FRONTEND_PKG_ROOT =
  process.env.ASKPDF_PLAYWRIGHT_ROOT ||
  (fs.existsSync(path.join(REPO_ROOT, 'frontend/node_modules/@playwright/test'))
    ? path.join(REPO_ROOT, 'frontend')
    : '/Users/raghu/Documents/Projects/askPDF/frontend');
const requireFromFrontend = createRequire(path.join(FRONTEND_PKG_ROOT, 'package.json'));
const { chromium } = requireFromFrontend('@playwright/test');

const ART_DIR =
  process.env.ASKPDF_E2E_ARTIFACTS ||
  '/Users/raghu/askpdf-worktrees/tool-inventory-artifacts/screenshots';
const PDF =
  process.env.ASKPDF_E2E_PDF || path.join(REPO_ROOT, 'rag_service/tests/01030000000000.pdf');
const MODEL_CANDIDATES = (
  process.env.ASKPDF_E2E_LLM_MODEL || 'nvidia/nemotron-3.5-super-vl:free,particle-ai/particle-1:free'
).split(',');

async function pickModel(page) {
  for (const model of MODEL_CANDIDATES) {
    try {
      await selectLlmModel(page, model.trim());
      return model.trim();
    } catch (err) {
      console.warn('[tool-inventory-ui-proof] model failed:', model, err?.message || err);
    }
  }
  throw new Error(`No LLM model available from: ${MODEL_CANDIDATES.join(', ')}`);
}

async function setContextWindow(page, model, tokens, threadId, threadName) {
  await page.evaluate((value) => {
    window.localStorage.setItem('last_context_window', String(value));
  }, tokens);
  await focusThreadInUi(page, threadName);
  if (model) {
    await selectLlmModel(page, model);
  }
  await waitForComposerReady(page);
  const btn = page.getByRole('button', { name: new RegExp(`^Context `) }).first();
  await btn.waitFor({ state: 'visible', timeout: 60_000 });
  const label = await btn.innerText();
  if (!label.replace(/,/g, '').includes(String(tokens))) {
    throw new Error(`Context button does not show ${tokens}: got "${label}"`);
  }
}

async function openPromptSettings(page) {
  const gear = page
    .getByRole('button', { name: /AI prompt settings/i })
    .or(page.locator('button[title*="AI prompt settings"]'))
  await gear.first().click({ timeout: 30_000 });
  await page.getByRole('dialog').getByText('AI Prompt Settings').waitFor({ timeout: 30_000 });
}

function apiChat(admin, threadId, payload) {
  const args = [
    '-sS',
    '-m',
    '900',
    '-X',
    'POST',
    `${DEFAULT_API_URL}/api/threads/${threadId}/chat`,
    '-H',
    `Authorization: Bearer ${admin}`,
    '-H',
    'Content-Type: application/json',
    '-d',
    JSON.stringify(payload),
  ];
  const result = spawnSync('curl', args, { encoding: 'utf8', maxBuffer: 20 * 1024 * 1024 });
  if (result.status !== 0) {
    throw new Error(`chat failed: ${(result.stderr || result.stdout || '').slice(0, 800)}`);
  }
  let body;
  try {
    body = JSON.parse(result.stdout);
  } catch {
    throw new Error(`chat returned non-JSON: ${result.stdout.slice(0, 800)}`);
  }
  if (body.detail) {
    throw new Error(`chat error: ${JSON.stringify(body.detail).slice(0, 800)}`);
  }
  return body;
}

async function focusThreadInUi(page, threadName) {
  await page.goto(DEFAULT_BASE_URL, { waitUntil: 'domcontentloaded', timeout: 120_000 });
  const row = page.getByText(threadName, { exact: false }).first();
  await row.waitFor({ state: 'visible', timeout: 120_000 });
  await row.click({ force: true });
  await page.waitForTimeout(1500);
}

function metricsFromDebug(debug) {
  const trace = debug?.trace || {};
  const metrics = trace?.metrics || debug?.metrics || {};
  return {
    llm_token_count_prompt: Number(metrics.llm_token_count_prompt || 0),
    llm_token_count_completion: Number(metrics.llm_token_count_completion || 0),
    llm_token_count_total: Number(metrics.llm_token_count_total || 0),
  };
}

async function openDebugTraceWorkspace(page) {
  const debugTab = page.getByRole('tab', { name: /^Debug Trace$/i });
  await debugTab.waitFor({ state: 'visible', timeout: 30_000 });
  await debugTab.click();
  await page.getByLabel('Open traces').waitFor({ state: 'visible', timeout: 120_000 });
  await page.waitForTimeout(1000);
  const promptTokens = page.getByText(/^Prompt tokens/i).first();
  if (!(await promptTokens.isVisible({ timeout: 5000 }).catch(() => false))) {
    const promptBtn = page.getByRole('button', { name: /^Prompt$/i }).first();
    if (await promptBtn.isVisible({ timeout: 5000 }).catch(() => false)) {
      await promptBtn.click();
      await page.waitForTimeout(500);
    }
  }
}

async function runPhase({ page, guard, admin, threadId, contextWindow, slug, questions, model, threadName }) {
  await setContextWindow(page, model, contextWindow, threadId, threadName);
  await guard.capture(page, `toolinv-${slug}-context-${contextWindow}.png`, { requireChatWorkspace: true });

  const results = [];
  for (const question of questions) {
    await chatRateLimitPause();
    const chatBody = await apiChat(admin, threadId, {
      thread_id: threadId,
      question,
      llm_model: model,
      use_web_search: false,
      use_reranker: true,
      context_window: contextWindow,
    });
    const preview = String(chatBody.answer || chatBody.final_answer || '').slice(0, 240);
    const runId = chatBody.agent_run_id || chatBody.run_id || null;
    let metrics = {};
    if (runId) {
      const run = await apiJson(`${DEFAULT_API_URL}/api/agent-runs/${runId}?thread_id=${threadId}`, { admin });
      const debug = run?.agent_run?.debug || run?.debug || {};
      metrics = metricsFromDebug(debug);
    }
    await focusThreadInUi(page, threadName);
    await waitForComposerReady(page);
    const openTrace = page.getByText(/Open trace ·/i).last();
    await openTrace.waitFor({ state: 'visible', timeout: 180_000 });
    await openTrace.scrollIntoViewIfNeeded();
    await openTrace.click();
    await openDebugTraceWorkspace(page);
    await guard.capture(page, `toolinv-${slug}-debug-${contextWindow}-${results.length + 1}.png`, {
      requireChatWorkspace: false,
    });
    results.push({ question, preview, run_id: runId, metrics });
    const chatTab = page.getByRole('tab', { name: /^Chat$/i }).first();
    if (await chatTab.isVisible({ timeout: 3000 }).catch(() => false)) {
      await chatTab.click();
      await waitForComposerReady(page);
    }
  }
  return results;
}

async function main() {
  if (!fs.existsSync(PDF)) throw new Error(`PDF missing: ${PDF}`);
  fs.mkdirSync(ART_DIR, { recursive: true });
  const admin = loadEnvAdmin();
  const guard = new ScreenshotGuard(ART_DIR);
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: REAL_UI_VIEWPORT });

  const meta = {
    base_url: DEFAULT_BASE_URL,
    pdf: PDF,
    tested_sha: null,
    model: null,
    phases: [],
    captured_at: new Date().toISOString(),
  };

  try {
    const boot = await bootstrapProjectThreadViaApi(page, `toolinv-${Date.now()}`);
    const threadId = boot.threadId;
    const threadName = boot.threadName;
    const model = await pickModel(page);
    meta.model = model;
    await uploadPdfThroughUi(page, PDF);
    await waitForComposerReady(page);

    await openPromptSettings(page);
    await guard.capture(page, 'toolinv-prompt-settings-catalog.png', { requireChatWorkspace: false });
    await page.getByRole('button', { name: /^Cancel$/i }).click({ timeout: 5000 }).catch(() => page.keyboard.press('Escape'));
    await waitForComposerReady(page);

    const phase128 = await runPhase({
      page,
      guard,
      admin,
      threadId,
      model,
      threadName,
      contextWindow: 131_072,
      slug: '128k',
      questions: [
        'What tools can you use? List the main categories in one short paragraph.',
        'Use get_thread_shape and tell me how many pages are in this thread PDF.',
      ],
    });
    meta.phases.push({ context_window: 131072, runs: phase128 });

    const phase8k = await runPhase({
      page,
      guard,
      admin,
      threadId,
      model,
      threadName,
      contextWindow: 8192,
      slug: '8k',
      questions: ['What tools can you use? Reply briefly.'],
    });
    meta.phases.push({ context_window: 8192, runs: phase8k });

    const shaProc = await import('node:child_process');
    const testedSha = shaProc
      .spawnSync('git', ['rev-parse', 'HEAD'], { cwd: REPO_ROOT, encoding: 'utf8' })
      .stdout.trim();
    meta.tested_sha = testedSha;
    const metaPath = path.join(ART_DIR, 'tool-inventory-ui-proof.json');
    fs.writeFileSync(metaPath, `${JSON.stringify(meta, null, 2)}\n`);
    console.log('[tool-inventory-ui-proof] ok', metaPath);
    console.log('[tool-inventory-ui-proof] screenshots', guard.paths.length);
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error('[tool-inventory-ui-proof] FAILED', err);
  process.exit(1);
});