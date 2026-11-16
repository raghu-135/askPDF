#!/usr/bin/env node
/**
 * One continuous real-UI proof for the clarification HITL card (PR #109).
 * Uses scripts/cloud-agent-e2e-lib.mjs (PR #110).
 */
import { execSync } from 'node:child_process';
import crypto from 'node:crypto';
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

function md5File(filePath) {
  const data = fs.readFileSync(filePath);
  return crypto.createHash('md5').update(data).digest('hex');
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

  const decisionPanel = page
    .locator('div')
    .filter({ has: page.getByText('I need a bit more clarification') })
    .filter({ has: page.getByRole('button', { name: /Something else/i }) })
    .last();
  const optionFields = decisionPanel.locator('textarea.MuiInputBase-inputMultiline');
  const optionCount = await optionFields.count();
  let optionText = '';
  let optionIndex = 0;
  for (let index = 0; index < optionCount; index += 1) {
    const candidate = (await optionFields.nth(index).inputValue()).trim();
    if (/compare|MORAN|DINER|them/i.test(candidate)) {
      optionText = candidate;
      optionIndex = index;
      break;
    }
  }
  if (!optionText) {
    throw new Error('No clarification option matching the uploaded PDF was found');
  }
  await optionFields.nth(optionIndex).click();
  await guard.capture(page, 'clarification-06-option-field.png', { requireChatWorkspace: true });

  await decisionPanel.locator('button[aria-label^="Send "]').nth(optionIndex).click();
  const composer = page.getByTestId('chat-composer-input');
  await composer.waitFor({ state: 'visible', timeout: 30_000 });
  const seeded = (await composer.inputValue()).trim();
  if (!seeded.includes(optionText.slice(0, Math.min(24, optionText.length)))) {
    throw new Error(`Composer was not seeded from option; expected fragment of ${JSON.stringify(optionText)}`);
  }
  await guard.capture(page, 'clarification-07-composer-seeded.png', { requireChatWorkspace: true });

  const edited = `${seeded.replace(/\s+$/, '')} (edited for send)`;
  await composer.fill(edited);
  await guard.capture(page, 'clarification-08-composer-edited.png', { requireChatWorkspace: true });

  await chatRateLimitPause();
  await typeAndSendChat(page, edited);
  await waitOutOfSending(page);
  await waitForComposerReady(page);
  const listText = (await page.getByTestId('chat-message-list').innerText()).trim();
  const lastAnswer = listText;
  if (/attached documents do not address/i.test(lastAnswer)) {
    throw new Error(`Got scope-decline answer after clarified send: ${lastAnswer.slice(0, 240)}`);
  }
  if (!/(MORAN|DINER|encoder|architecture|layer|attention)/i.test(lastAnswer)) {
    throw new Error(`Answer does not look PDF-grounded: ${lastAnswer.slice(-400)}`);
  }
  await guard.capture(page, 'clarification-09-grounded-answer.png', { requireChatWorkspace: true });

  await chatRateLimitPause();
  await page.waitForTimeout(4000);
  await waitForComposerReady(page);
  await typeAndSendChat(page, QUESTION);
  await waitClarificationCard(page);
  const cancelPanel = page
    .locator('div')
    .filter({ has: page.getByText('I need a bit more clarification') })
    .filter({ has: page.getByRole('button', { name: /Something else/i }) })
    .last();
  await guard.capture(page, 'clarification-10-card-before-cancel.png', { requireChatWorkspace: true });
  const closeBtn = cancelPanel.getByRole('button', { name: /Close decision panel/i });
  await closeBtn.scrollIntoViewIfNeeded();
  await closeBtn.evaluate((node) => node.click());
  await waitForComposerReady(page);
  const reseed = await composer.inputValue();
  if (!reseed.includes(QUESTION)) {
    throw new Error(`Cancel did not re-seed composer with ${JSON.stringify(QUESTION)}; got ${JSON.stringify(reseed)}`);
  }
  await guard.capture(page, 'clarification-11-cancel-reseed.png', { requireChatWorkspace: true });

  const screenshotMd5 = Object.fromEntries(
    guard.paths.map((p) => [path.basename(p), md5File(p)]),
  );

  const meta = {
    tested_sha: sha,
    model: MODEL,
    question: QUESTION,
    picked_option: optionText,
    sent_question: edited,
    assistant_answer_preview: lastAnswer.slice(0, 280),
    custom_question: CUSTOM,
    screenshots: guard.paths,
    screenshot_md5: screenshotMd5,
  };
  fs.writeFileSync(path.join(DEFAULT_ART_DIR, 'clarification-verify-metadata.json'), JSON.stringify(meta, null, 2));
  await browser.close();
  console.log(JSON.stringify(meta, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});