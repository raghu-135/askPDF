#!/usr/bin/env node
/**
 * Real-UI proof for tool-approval bottom sheet (1440×900, distinct screenshots).
 */
import { spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import {
  DEFAULT_ART_DIR,
  DEFAULT_BASE_URL,
  REAL_UI_VIEWPORT,
  ScreenshotGuard,
  chatRateLimitPause,
  ensureProjectAndThread,
  selectLlmModel,
  waitForComposerReady,
} from './cloud-agent-e2e-lib.mjs';

const REPO_ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const requireFromFrontend = createRequire(path.join(REPO_ROOT, 'frontend/package.json'));
const { chromium } = requireFromFrontend('@playwright/test');

const MODEL = process.env.ASKPDF_E2E_LLM_MODEL || 'inclusionai/ling-3.2-flash:free';
const ART_DIR = DEFAULT_ART_DIR;

function gitSha() {
  const r = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: REPO_ROOT, encoding: 'utf8' });
  if (r.status !== 0) return 'unknown';
  return r.stdout.trim();
}

async function selectRouterAgentWorkflow(page) {
  await page.getByRole('button', { name: /AI prompt settings for this thread/i }).click();
  const dialog = page.getByRole('dialog');
  await dialog.waitFor({ timeout: 30000 });
  await dialog.getByRole('combobox', { name: /Agent workflow/i }).click();
  await page.getByRole('option', { name: /Router Agent/i }).click();
  await dialog.getByRole('button', { name: /^Save$/i }).scrollIntoViewIfNeeded();
  await dialog.getByRole('button', { name: /^Save$/i }).click();
  await dialog.waitFor({ state: 'hidden', timeout: 60000 });
}

async function setWebSearchAsk(page) {
  const labels = ['Internet Search Off', 'Internet Search On', 'Ask me every time before internet search'];
  const btn = page.getByRole('button', { name: new RegExp(labels.join('|')) }).first();
  await btn.scrollIntoViewIfNeeded();
  for (let i = 0; i < labels.length; i += 1) {
    if ((await btn.getAttribute('aria-label')) === labels[2]) return;
    await btn.click();
    await page.waitForTimeout(350);
  }
}

async function dismissClarificationIfPresent(page) {
  const original = page.getByRole('button', { name: /Original question/i }).first();
  if (await original.isVisible({ timeout: 1500 }).catch(() => false)) {
    await original.click();
  }
}

async function waitForApprovalSheet(page) {
  const sheet = page.getByTestId('tool-approval-bottom-sheet');
  const title = page.getByText(/Allow a web search\?/i).first();
  await title.waitFor({ state: 'visible', timeout: 180000 });
  await sheet.waitFor({ state: 'visible', timeout: 5000 });
  await page.getByRole('button', { name: /Approve once/i }).scrollIntoViewIfNeeded();
}

async function dragSheetHandle(page, deltaY) {
  const handle = page.getByRole('separator', { name: /Resize approval sheet/i }).first();
  const box = await handle.boundingBox();
  if (!box) throw new Error('approval sheet resize handle not found');
  const startX = box.x + box.width / 2;
  const startY = box.y + box.height / 2;
  await page.mouse.move(startX, startY);
  await page.mouse.down();
  await page.mouse.move(startX, startY - deltaY, { steps: 12 });
  await page.mouse.up();
  await page.waitForTimeout(400);
}

async function main() {
  fs.mkdirSync(ART_DIR, { recursive: true });
  const guard = new ScreenshotGuard(ART_DIR);
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: REAL_UI_VIEWPORT });
  const page = await context.newPage();

  try {
    const testedSha = gitSha();
    await page.goto(DEFAULT_BASE_URL, { waitUntil: 'domcontentloaded', timeout: 120000 });
    await ensureProjectAndThread(page);
    await selectLlmModel(page, MODEL);
    await selectRouterAgentWorkflow(page);
    await selectLlmModel(page, MODEL);
    await setWebSearchAsk(page);
    await waitForComposerReady(page);

    await chatRateLimitPause();
    const longQuery = [
      'Use the internet search tool to find authoritative documentation covering',
      'PDF text extraction quality, OCR fallback behavior, Unicode normalization,',
      'and chunk boundary stability for RAG indexing pipelines in enterprise document search.',
      'Reply with one sentence.',
    ].join(' ');
    const composer = page.getByTestId('chat-composer-input');
    await composer.fill(longQuery);
    await page.keyboard.press('Enter');
    await dismissClarificationIfPresent(page);
    await waitForApprovalSheet(page);

    await guard.capture(page, 'hitl-01-sheet-compact.png');
    await dragSheetHandle(page, 220);
    await guard.capture(page, 'hitl-02-sheet-tall.png');

    const scrollRegion = page.getByTestId('review-decision-scroll');
    await scrollRegion.evaluate((el) => { el.scrollTop = el.scrollHeight; });
    await page.waitForTimeout(300);
    await guard.capture(page, 'hitl-03-sheet-content-scrolled.png');

    await dragSheetHandle(page, -180);
    const pendingPlaceholder = page.getByPlaceholder(/Approve or skip to continue/i);
    await pendingPlaceholder.waitFor({ state: 'visible', timeout: 10000 });
    await pendingPlaceholder.scrollIntoViewIfNeeded();
    await page.waitForTimeout(300);
    await guard.capture(page, 'hitl-04-composer-disabled.png');

    const debugTraceTab = page.getByRole('tab', { name: /^Debug Trace/i }).first();
    await debugTraceTab.waitFor({ state: 'visible', timeout: 30000 });
    await debugTraceTab.click();
    await page.getByTestId('debug-human-review-banner').waitFor({ state: 'visible', timeout: 60000 });
    await page.getByRole('button', { name: /Approve once/i }).last().scrollIntoViewIfNeeded();
    await guard.capture(page, 'hitl-05-debug-trace-review-panel.png');

    const threadTab = page.getByRole('tab', { name: /^Thread$/i }).first();
    await threadTab.waitFor({ state: 'visible', timeout: 30000 });
    await threadTab.click();
    await page.getByTestId('tool-approval-bottom-sheet').getByRole('button', { name: /Approve once/i }).click({ force: true });
    try {
      await page.getByTestId('tool-approval-bottom-sheet').waitFor({ state: 'hidden', timeout: 360000 });
    } catch {
      await page.getByTestId('tool-approval-bottom-sheet').getByRole('button', { name: /^Skip$/i }).click({ force: true });
      await page.getByTestId('tool-approval-bottom-sheet').waitFor({ state: 'hidden', timeout: 120000 });
    }
    await waitForComposerReady(page);
    await page.getByTestId('chat-composer-input').waitFor({ state: 'visible', timeout: 300000 });
    await page.waitForTimeout(2000);
    await guard.capture(page, 'hitl-06-composer-enabled-after-approve.png');

    const screenshotMeta = guard.paths.map((p) => ({
      path: p,
      md5: crypto.createHash('md5').update(fs.readFileSync(p)).digest('hex'),
      basename: path.basename(p),
    }));
    const metadata = {
      tested_sha: testedSha,
      model: MODEL,
      base_url: DEFAULT_BASE_URL,
      screenshots: screenshotMeta,
      captured_at: new Date().toISOString(),
    };
    fs.writeFileSync(path.join(ART_DIR, 'hitl-bottom-sheet-verify-metadata.json'), `${JSON.stringify(metadata, null, 2)}\n`);
    console.log('[hitl-bottom-sheet-verify] tested_sha', testedSha);
    for (const row of screenshotMeta) {
      console.log(`  ${row.path} md5=${row.md5}`);
    }
  } finally {
    await context.close();
    await browser.close();
  }
}

main().catch((err) => {
  console.error('[hitl-bottom-sheet-verify] FAILED:', err);
  process.exit(1);
});