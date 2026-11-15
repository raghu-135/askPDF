#!/usr/bin/env node
/**
 * One continuous real-UI proof for the clarification HITL card (PR #109).
 * Uses scripts/cloud-agent-e2e-lib.mjs (PR #110).
 */
import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import pathUtil from 'node:path';
import { fileURLToPath } from 'node:url';

const requireFromFrontend = createRequire(pathUtil.join(pathUtil.dirname(fileURLToPath(import.meta.url)), '../frontend/package.json'));
const { chromium } = requireFromFrontend('@playwright/test');
import {
  DEFAULT_API_URL,
  DEFAULT_ART_DIR,
  REAL_UI_VIEWPORT,
  REPO_ROOT,
  ScreenshotGuard,
  chatRateLimitPause,
  ensureProjectAndThread,
  loadEnvAdmin,
  selectLlmModel,
  typeAndSendChat,
  uploadPdfThroughUi,
  waitForComposerReady,
  waitForNoLoadingShell,
  waitForThreadWorkspaceReady,
} from './cloud-agent-e2e-lib.mjs';

const MODEL = process.env.ASKPDF_E2E_LLM_MODEL || 'inclusionai/ling-3.2-flash:free';
const PDF = process.env.ASKPDF_E2E_PDF || path.join(REPO_ROOT, 'rag_service/tests/01030000000000.pdf');
const QUESTION = process.env.ASKPDF_CLARIFY_QUESTION || 'Compare them.';
const CUSTOM = 'Summarize the architecture section in one paragraph.';

async function waitClarificationCard(page) {
  await page.getByText('I need a bit more clarification').waitFor({ timeout: 600_000 });
  await page.getByRole('button', { name: /Something else/i }).waitFor({ timeout: 60_000 });
}

async function waitOutOfSending(page) {
  const sending = page.getByPlaceholder(/Sending/i);
  if (await sending.isVisible({ timeout: 120_000 }).catch(() => false)) {
    await sending.waitFor({ state: 'hidden', timeout: 900_000 });
  }
}

async function main() {
  const sha = execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim();
  const admin = loadEnvAdmin();
  const guard = new ScreenshotGuard(DEFAULT_ART_DIR);
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: REAL_UI_VIEWPORT });
  await page.addInitScript(({ token, model }) => {
    window.localStorage.setItem('auth_token', token);
    window.localStorage.setItem('last_llm_model', model);
  }, { token: admin, model: MODEL });

  await ensureProjectAndThread(page);
  await waitForThreadWorkspaceReady(page);
  await guard.capture(page, 'clarification-01-thread-ready.png', { requireChatWorkspace: true });

  await selectLlmModel(page, MODEL);
  await guard.capture(page, 'clarification-02-model-selected.png', { requireChatWorkspace: true });

  await uploadPdfThroughUi(page, PDF);
  await page.getByRole('tab', { name: /Thread/i }).first().click().catch(() => undefined);
  await waitForComposerReady(page);
  await guard.capture(page, 'clarification-03-indexed-ready.png', { requireChatWorkspace: true });

  await typeAndSendChat(page, QUESTION);
  await waitClarificationCard(page);
  await guard.capture(page, 'clarification-04-card.png', { requireChatWorkspace: true });

  await page.getByRole('button', { name: /Something else/i }).click();
  await page.getByRole('textbox', { name: 'Your response' }).fill(CUSTOM);
  await guard.capture(page, 'clarification-05-custom-box.png', { requireChatWorkspace: true });
  await page.getByRole('button', { name: 'Cancel custom response' }).click();

  const option = page.getByRole('button').filter({ hasText: /Compare|MORAN/i }).first();
  const optionLabel = (await option.innerText()).trim();
  await option.click();
  const sendingAfterOption = page.getByPlaceholder(/Sending/i);
  await sendingAfterOption.waitFor({ state: 'visible', timeout: 120_000 });
  await guard.capture(page, 'clarification-06-option-sending.png', { requireChatWorkspace: true, allowLoadingShell: true });
  await waitOutOfSending(page);
  await waitForComposerReady(page);
  await guard.capture(page, 'clarification-07-option-answered.png', { requireChatWorkspace: true });

  await chatRateLimitPause();
  await page.waitForTimeout(4000);
  await waitForComposerReady(page);
  await typeAndSendChat(page, QUESTION);
  await waitClarificationCard(page);
  await guard.capture(page, 'clarification-08-card-before-cancel.png', { requireChatWorkspace: true });
  await page.getByLabel('Close decision panel').click();
  await waitForComposerReady(page);
  const composer = page.getByTestId('chat-composer-input');
  const reseed = await composer.inputValue();
  if (!reseed.includes(QUESTION)) {
    throw new Error(`Cancel did not re-seed composer with ${JSON.stringify(QUESTION)}; got ${JSON.stringify(reseed)}`);
  }
  await guard.capture(page, 'clarification-09-cancel-reseed.png', { requireChatWorkspace: true });

  const meta = {
    tested_sha: sha,
    model: MODEL,
    question: QUESTION,
    picked_option: optionLabel,
    custom_question: CUSTOM,
    screenshots: guard.paths,
  };
  fs.writeFileSync(path.join(DEFAULT_ART_DIR, 'clarification-verify-metadata.json'), JSON.stringify(meta, null, 2));
  await browser.close();
  console.log(JSON.stringify(meta, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});