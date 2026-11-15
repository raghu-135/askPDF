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

async function setBrowserAskMode(page) {
  await page.evaluate(() => {
    localStorage.setItem('last_browser_access_mode', 'ask');
    window.dispatchEvent(new CustomEvent('askpdf:browser-access-mode', { detail: 'ask' }));
  });
  const askBtn = page.getByRole('button', { name: /^Ask before the agent uses the workbench browser$/i });
  for (let i = 0; i < 6; i += 1) {
    if (await askBtn.isVisible({ timeout: 1500 }).catch(() => false)) {
      return;
    }
    const toggle = page
      .getByRole('button', {
        name: /Browser agent off|Browser agent on|Ask before the agent uses the workbench browser/i,
      })
      .first();
    await toggle.click();
    await page.waitForTimeout(300);
  }
}

async function ensureAutoWorkflow(page) {
  const settings = page.getByRole('button', { name: /AI prompt settings for this thread/i }).first();
  await settings.click();
  const dialog = page.getByRole('dialog');
  await dialog.waitFor({ timeout: 15000 });
  const workflow = dialog.getByRole('combobox', { name: /^Agent workflow/i });
  await workflow.click();
  const auto = page.getByRole('option', { name: /^Auto$/i }).first();
  await auto.click();
  await dialog.getByRole('button', { name: /^Save$/i }).click();
  await dialog.waitFor({ state: 'hidden', timeout: 15000 });
}

async function clickApproveOnce(page) {
  const footer = page.getByTestId('review-decision-footer');
  await footer.waitFor({ state: 'visible', timeout: 420_000 });
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
    await guard.capture(page, 'auto-browser-01-ask-mode-1440x900.png', { requireChatWorkspace: true });
    await uploadPdfThroughUi(page, PDF_PATH);
    await waitForComposerReady(page);
    await guard.capture(page, 'auto-browser-02-pdf-uploaded-1440x900.png', { requireChatWorkspace: true });

    await chatRateLimitPause();
    await typeAndSendChat(
      page,
      'Use the browser to find papers similar to the attached document. Search scholarly sources and cite real paper titles and URLs you find.',
    );
    await clickApproveOnce(page);
    await guard.capture(page, 'auto-browser-03-browser-approval-1440x900.png', { requireChatWorkspace: true });
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