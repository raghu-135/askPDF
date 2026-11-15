#!/usr/bin/env node
/**
 * Auto workflow + browser workbench proof (uses scripts/cloud-agent-e2e-lib.mjs from PR #110).
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
  ensureProjectAndThread,
  selectLlmModel,
  typeAndSendChat,
  uploadPdfThroughUi,
  waitForAssistantAnswer,
  waitForComposerReady,
  setBrowserAskMode,
} from './cloud-agent-e2e-lib.mjs';

const requireFromFrontend = createRequire(path.join(REPO_ROOT, 'frontend/package.json'));
const { chromium } = requireFromFrontend('@playwright/test');

const MODEL = process.env.ASKPDF_E2E_LLM_MODEL || 'google/gemma-4-e4b';
const FALLBACK_MODEL = 'inclusionai/ling-3.2-flash:free';
const PDF_PATH = process.env.ASKPDF_E2E_PDF || path.join(REPO_ROOT, 'rag_service/tests/01030000000000.pdf');
const ART_DIR = DEFAULT_ART_DIR;

function gitSha() {
  const r = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: REPO_ROOT, encoding: 'utf8' });
  return r.status === 0 ? r.stdout.trim() : 'unknown';
}

async function ensureAutoWorkflow(page) {
  const workflow = page.getByRole('combobox', { name: /^Agent workflow/i }).first();
  if (!(await workflow.isVisible({ timeout: 8000 }).catch(() => false))) {
    const threadTab = page.getByRole('tab', { name: /^Thread$/i }).first();
    if (await threadTab.isVisible({ timeout: 3000 }).catch(() => false)) {
      await threadTab.click();
    }
    await workflow.waitFor({ state: 'visible', timeout: 30_000 });
  }
  const current = (await workflow.textContent()) || '';
  if (/\bauto\b/i.test(current)) return;
  await workflow.click();
  const auto = page.getByRole('option', { name: /auto_rag_agent|^Auto\b/i }).first();
  await auto.click({ timeout: 15_000 });
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
  await footer.waitFor({ state: 'visible', timeout: 30_000 });
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

async function selectModel(page) {
  try {
    await selectLlmModel(page, MODEL);
    return MODEL;
  } catch (error) {
    console.warn(`[auto-browser-proof] model ${MODEL} unavailable, falling back: ${error}`);
    await selectLlmModel(page, FALLBACK_MODEL);
    return FALLBACK_MODEL;
  }
}

async function main() {
  fs.mkdirSync(ART_DIR, { recursive: true });
  const guard = new ScreenshotGuard(ART_DIR);
  const browser = await chromium.launch({
    headless: process.env.ASKPDF_E2E_HEADLESS !== '0',
  });
  const context = await browser.newContext({ viewport: REAL_UI_VIEWPORT });
  const page = await context.newPage();
  const testedSha = gitSha();

  try {
    await page.goto(DEFAULT_BASE_URL, { waitUntil: 'domcontentloaded', timeout: 120_000 });
    await ensureProjectAndThread(page);
    const model = await selectModel(page);
    await ensureAutoWorkflow(page);
    await setBrowserAskMode(page);
    const askControl = page.getByRole('button', { name: /^Ask before the agent uses the workbench browser$/i });
    await askControl.waitFor({ state: 'visible', timeout: 15_000 });
    await guard.capture(page, 'auto-browser-01-ask-mode-1440x900.png', { requireChatWorkspace: true });
    await uploadPdfThroughUi(page, PDF_PATH);
    await waitForComposerReady(page);
    await guard.capture(page, 'auto-browser-02-pdf-uploaded-1440x900.png', { requireChatWorkspace: true });

    await chatRateLimitPause();
    await typeAndSendChat(
      page,
      'Use the browser to find papers similar to the attached document. Search scholarly sources and cite real paper titles and URLs you find.',
    );
    const approvalFooter = await waitForApprovalFooter(page);
    await approvalFooter.scrollIntoViewIfNeeded();
    await guard.capture(page, 'auto-browser-03-browser-approval-1440x900.png', { requireChatWorkspace: false });
    await clickApproveOnce(page);
    const browserTab = page.getByRole('tab', { name: /^Browser$/i }).first();
    await browserTab.click();
    await page.waitForTimeout(45_000);
    await guard.capture(page, 'auto-browser-04-browser-tab-1440x900.png');
    const chatTab = page.getByRole('tab', { name: /^Chat$/i }).first();
    if (await chatTab.isVisible({ timeout: 5000 }).catch(() => false)) {
      await chatTab.click();
    }
    const answer = await waitForAssistantAnswer(page);
    if (/technical error|do not address this question/i.test(answer)) {
      throw new Error(`Unexpected assistant answer: ${answer.slice(0, 240)}`);
    }
    await guard.capture(page, 'auto-browser-05-similar-papers-answer-1440x900.png', { requireChatWorkspace: true });

    await chatRateLimitPause();
    await typeAndSendChat(page, 'What tools do you have access to in this thread? List the main ones briefly.');
    const toolsAnswer = await waitForAssistantAnswer(page);
    if (/do not address this question/i.test(toolsAnswer)) {
      throw new Error(`Tool-capability answer was document-scope declined: ${toolsAnswer.slice(0, 240)}`);
    }
    await guard.capture(page, 'auto-browser-06-tools-capability-answer-1440x900.png', { requireChatWorkspace: true });

    const metadata = {
      tested_sha: testedSha,
      model,
      assistant_preview: answer.slice(0, 280),
      tools_answer_preview: toolsAnswer.slice(0, 280),
      screenshots: guard.paths.map((p) => ({
        path: p,
        basename: path.basename(p),
      })),
    };
    fs.writeFileSync(path.join(ART_DIR, 'auto-browser-proof-metadata.json'), `${JSON.stringify(metadata, null, 2)}\n`);
    console.log('[auto-browser-proof] OK', JSON.stringify(metadata, null, 2));
  } finally {
    await context.close();
    await browser.close();
  }
}

main().catch((error) => {
  console.error('[auto-browser-proof] FAILED', error);
  process.exit(1);
});