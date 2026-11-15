#!/usr/bin/env node
/**
 * End-to-end real-UI proof for Cloud Agents (stack required unless --prove-screenshot-guard).
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
  assertChatWorkspaceVisible,
  chatRateLimitPause,
  ensureProjectAndThread,
  openUploadedDocumentTab,
  proveScreenshotGuardFailsOnDuplicate,
  selectLlmModel,
  typeAndSendChat,
  uploadPdfThroughUi,
  waitForAssistantAnswer,
  waitForComposerReady,
} from './cloud-agent-e2e-lib.mjs';

const requireFromFrontend = createRequire(path.join(REPO_ROOT, 'frontend/package.json'));
const { chromium } = requireFromFrontend('@playwright/test');

const MODEL = process.env.ASKPDF_E2E_LLM_MODEL || 'inclusionai/ling-3.2-flash:free';
const PDF_PATH = process.env.ASKPDF_E2E_PDF || path.join(REPO_ROOT, 'rag_service/tests/01030000000000.pdf');
const ART_DIR = DEFAULT_ART_DIR;

function gitSha() {
  const r = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: REPO_ROOT, encoding: 'utf8' });
  if (r.status !== 0) return 'unknown';
  return r.stdout.trim();
}

async function runRealUiFlow() {
  if (!fs.existsSync(PDF_PATH)) {
    throw new Error(`PDF not found: ${PDF_PATH}`);
  }

  fs.mkdirSync(ART_DIR, { recursive: true });
  const guard = new ScreenshotGuard(ART_DIR);

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: REAL_UI_VIEWPORT });
  const page = await context.newPage();

  try {
    await page.goto(DEFAULT_BASE_URL, { waitUntil: 'domcontentloaded', timeout: 120_000 });
    await page.waitForTimeout(1500);
    await guard.capture(page, 'real-ui-01-landing.png');

    const { threadName } = await ensureProjectAndThread(page);
    await guard.capture(page, 'real-ui-02-thread-workspace.png');
    await assertChatWorkspaceVisible(page);

    await selectLlmModel(page, MODEL);
    await guard.capture(page, 'real-ui-03-model-selected.png');
    const uploadedName = await uploadPdfThroughUi(page, PDF_PATH);
    await openUploadedDocumentTab(page, uploadedName);
    await guard.capture(page, 'real-ui-04-document-workspace.png');
    const chatTab = page.getByRole('tab', { name: /^Chat$/i }).first();
    if (await chatTab.isVisible({ timeout: 5000 }).catch(() => false)) {
      await chatTab.click();
    }
    await waitForComposerReady(page);
    await guard.capture(page, 'real-ui-05-sources-indexed-chat.png');

    await chatRateLimitPause();
    const question = 'What is the title of this PDF? Reply in one short sentence and cite the document.';
    await typeAndSendChat(page, question);
    const answerPreview = (await waitForAssistantAnswer(page)).slice(0, 200);
    console.log('[real-ui-verify] assistant preview:', answerPreview);
    await guard.capture(page, 'real-ui-06-assistant-answer.png');

    if (threadName) {
      const threadLink = page.getByText(threadName, { exact: false }).first();
      if (await threadLink.isVisible({ timeout: 3000 }).catch(() => false)) {
        await threadLink.click();
        await waitForComposerReady(page);
        await guard.capture(page, 'real-ui-07-sidebar-thread.png');
      }
    }

    console.log('[real-ui-verify] tested_sha', gitSha());
    console.log('[real-ui-verify] model', MODEL);
    console.log('[real-ui-verify] screenshots:');
    for (const p of guard.paths) {
      console.log(' ', p);
    }
  } finally {
    await context.close();
    await browser.close();
  }
}

function runProveGuard() {
  const msg = proveScreenshotGuardFailsOnDuplicate(ART_DIR);
  console.log('[real-ui-verify] screenshot guard rejected duplicate as expected');
  console.log('[real-ui-verify]', msg);
}

async function main() {
  const proveOnly = process.argv.includes('--prove-screenshot-guard');
  if (proveOnly) {
    runProveGuard();
    return;
  }
  await runRealUiFlow();
  if (process.argv.includes('--with-guard-self-test')) {
    runProveGuard();
  }
}

main().catch((err) => {
  console.error('[real-ui-verify] FAILED:', err);
  process.exit(1);
});