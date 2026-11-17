#!/usr/bin/env node
/**
 * Prove delegated browser tool approval shows the user question exactly once
 * (pending, after approve, after reload).
 */
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import {
  DEFAULT_ART_DIR,
  DEFAULT_BASE_URL,
  REPO_ROOT,
  REAL_UI_VIEWPORT,
  ScreenshotGuard,
  chatRateLimitPause,
  bootstrapProjectThreadViaApi,
  loadEnvAdmin,
  apiJson,
  selectLlmModel,
  typeAndSendChat,
  uploadPdfThroughUi,
  waitForComposerReady,
  waitForNoLoadingShell,
  waitOutOfSending,
  setBrowserAskMode,
} from './cloud-agent-e2e-lib.mjs';

const requireFromFrontend = createRequire(path.join(REPO_ROOT, 'frontend/package.json'));
const { chromium } = requireFromFrontend('@playwright/test');

const MODEL = 'inclusionai/ling-3.2-flash:free';
const PDF_PATH = process.env.ASKPDF_E2E_PDF || path.join(REPO_ROOT, 'rag_service/tests/01030000000000.pdf');
const ART_DIR = process.env.ASKPDF_ARTIFACT_DIR || '/opt/cursor/artifacts';
const QUESTION =
  'Use the browser workbench to search Google Scholar for papers related to the uploaded PDF. Stop at tool approval.';

function gitSha() {
  const r = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: REPO_ROOT, encoding: 'utf8' });
  return r.status === 0 ? r.stdout.trim() : 'unknown';
}

async function waitForServiceReady() {
  const apiBase = process.env.ASKPDF_API_URL || 'http://127.0.0.1:8000';
  const appBase = process.env.ASKPDF_E2E_BASE_URL || DEFAULT_BASE_URL;
  for (let attempt = 0; attempt < 90; attempt += 1) {
    try {
      const [apiOk, appOk] = await Promise.all([
        fetch(`${apiBase}/health`).then((r) => r.ok).catch(() => false),
        fetch(appBase).then((r) => r.ok).catch(() => false),
      ]);
      if (apiOk && appOk) return;
    } catch {
      // retry
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  throw new Error(`Services not ready (api=${apiBase}, app=${appBase})`);
}

async function ensureAutoWorkflow(page) {
  const workflow = page.getByRole('combobox', { name: /^Agent workflow/i }).first();
  await workflow.waitFor({ state: 'visible', timeout: 60_000 });
  const current = (await workflow.textContent()) || '';
  if (/\bauto\b/i.test(current)) return;
  await workflow.click();
  await page.getByRole('option', { name: /auto_rag_agent|^Auto\b/i }).first().click({ timeout: 15_000 });
  const save = page.getByRole('button', { name: /^Save$/i });
  if (await save.isVisible({ timeout: 2000 }).catch(() => false)) {
    await save.click();
    await page.getByRole('dialog').waitFor({ state: 'hidden', timeout: 15_000 }).catch(() => {});
  }
}

async function waitForApprovalFooter(page) {
  const footer = page.getByTestId('review-decision-footer');
  for (let attempt = 0; attempt < 120; attempt += 1) {
    if (await footer.isVisible({ timeout: 1000 }).catch(() => false)) {
      return footer;
    }
    await page.waitForTimeout(5000);
  }
  await footer.waitFor({ state: 'visible', timeout: 120_000 });
  return footer;
}

async function clickApproveOnce(page) {
  const footer = page.getByTestId('review-decision-footer');
  await footer.waitFor({ state: 'visible', timeout: 60_000 });
  const approve = footer.getByRole('button', { name: /^Approve once$/i }).first();
  await approve.evaluate((button) => {
    button.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
  });
  await footer.waitFor({ state: 'hidden', timeout: 240_000 }).catch(() => {});
}

async function countUserQuestions(page, snippet) {
  const list = page.getByTestId('chat-message-list');
  await list.waitFor({ state: 'visible', timeout: 60_000 });
  const needle = snippet.slice(0, 40).toLowerCase();
  return list.locator('li').evaluateAll((nodes, n) => {
    let hits = 0;
    for (const li of nodes) {
      const align = getComputedStyle(li).alignItems;
      if (align !== 'flex-end') continue;
      const text = (li.textContent || '').replace(/\s+/g, ' ').trim().toLowerCase();
      if (text.includes(n)) hits += 1;
    }
    return hits;
  }, needle);
}

function assertSingleQuestion(count, phase) {
  if (count !== 1) {
    throw new Error(`Expected exactly 1 user question during ${phase}, saw ${count}`);
  }
}

async function stampThreadId(page, threadId) {
  await page.evaluate((id) => {
    const prior = document.getElementById('e2e-thread-stamp');
    if (prior) prior.remove();
    const div = document.createElement('div');
    div.id = 'e2e-thread-stamp';
    div.textContent = `thread_id: ${id}`;
    Object.assign(div.style, {
      position: 'fixed',
      top: '8px',
      left: '8px',
      zIndex: '99999',
      background: 'rgba(0,0,0,0.72)',
      color: '#fff',
      padding: '6px 10px',
      font: '12px monospace',
      borderRadius: '4px',
      pointerEvents: 'none',
    });
    document.body.appendChild(div);
  }, threadId);
}

async function captureWithThread(guard, page, threadId, filename) {
  await stampThreadId(page, threadId);
  await guard.capture(page, filename, { requireChatWorkspace: false });
  fs.writeFileSync(
    path.join(ART_DIR, filename.replace(/\.png$/, '-thread.txt')),
    `${threadId}\n`,
  );
}

async function main() {
  fs.mkdirSync(ART_DIR, { recursive: true });
  const guard = new ScreenshotGuard(ART_DIR);
  const testedSha = gitSha();
  fs.writeFileSync(
    path.join(ART_DIR, 'delegate-question-dedupe-proof-head.txt'),
  `${testedSha}\n`,
  );

  const storagePath = path.join(ART_DIR, 'delegate-dedupe-storage.json');
  const browser = await chromium.launch({
    headless: process.env.ASKPDF_E2E_HEADLESS !== '0',
    args: ['--disable-dev-shm-usage', '--no-sandbox'],
  });
  const contextOptions = { viewport: REAL_UI_VIEWPORT };
  if (fs.existsSync(storagePath) && process.env.ASKPDF_E2E_IGNORE_STORAGE !== '1') {
    contextOptions.storageState = storagePath;
  }
  const context = await browser.newContext(contextOptions);
  const page = await context.newPage();

  try {
    await waitForServiceReady();
    const { threadId } = await bootstrapProjectThreadViaApi(page, `delegate-proof-${Date.now()}`);
    await selectLlmModel(page, MODEL);
    await ensureAutoWorkflow(page);
    await setBrowserAskMode(page);
    await uploadPdfThroughUi(page, PDF_PATH);
    await waitForComposerReady(page);

    await chatRateLimitPause();
    await typeAndSendChat(page, QUESTION);
    await waitForApprovalFooter(page);
    const pendingCount = await countUserQuestions(page, 'browser workbench');
    assertSingleQuestion(pendingCount, 'pending approval');
    await captureWithThread(guard, page, threadId, 'delegate-dedupe-01-pending-approval-1440x900.png');

    await clickApproveOnce(page);
    await waitOutOfSending(page);
    await waitForComposerReady(page);
    const afterApproveCount = await countUserQuestions(page, 'browser workbench');
    assertSingleQuestion(afterApproveCount, 'after approve');
    await captureWithThread(guard, page, threadId, 'delegate-dedupe-02-after-approve-1440x900.png');

    await context.storageState({ path: storagePath });
    await page.reload({ waitUntil: 'domcontentloaded', timeout: 120_000 });
    await waitForNoLoadingShell(page, { timeout: 300_000 });
    await page.getByTestId('chat-message-list').waitFor({ state: 'visible', timeout: 300_000 });
    await page
      .getByTestId('chat-message-list')
      .getByText(/browser workbench/i)
      .first()
      .waitFor({ state: 'visible', timeout: 120_000 });
    const afterReloadCount = await countUserQuestions(page, 'browser workbench');
    assertSingleQuestion(afterReloadCount, 'after reload');
    await captureWithThread(guard, page, threadId, 'delegate-dedupe-03-after-reload-1440x900.png');

    await chatRateLimitPause();
    await typeAndSendChat(page, 'What tools can you use?');
    await waitOutOfSending(page);
    await captureWithThread(guard, page, threadId, 'delegate-dedupe-04-follow-up-tools-1440x900.png');

    const metadata = {
      tested_sha: testedSha,
      thread_id: threadId,
      model: MODEL,
      question_snippet: QUESTION.slice(0, 80),
      counts: {
        pending: pendingCount,
        after_approve: afterApproveCount,
        after_reload: afterReloadCount,
      },
      screenshots: guard.paths.map((p) => path.basename(p)),
    };
    fs.writeFileSync(path.join(ART_DIR, 'delegate-question-dedupe-proof.json'), `${JSON.stringify(metadata, null, 2)}\n`);
    console.log('[delegate-dedupe-proof] OK', JSON.stringify(metadata, null, 2));
  } finally {
    await context.close();
    await browser.close();
  }
}

main().catch((error) => {
  console.error('[delegate-dedupe-proof] FAILED', error);
  process.exit(1);
});