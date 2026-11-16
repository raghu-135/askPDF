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
  DEFAULT_ART_DIR,
  REPO_ROOT,
  REAL_UI_VIEWPORT,
  ScreenshotGuard,
  assertGroundedPdfAnswer,
  assertNotScopeDecline,
  chatRateLimitPause,
  chatTranscriptText,
  clarificationPanel,
  ensureProjectAndThread,
  loadEnvAdmin,
  selectLlmModel,
  typeAndSendChat,
  uploadPdfThroughUi,
  waitClarificationCard,
  waitComposerReady,
  waitOutOfSending,
  waitRepeatClarifyOrGrounded,
  waitForThreadWorkspaceReady,
} from './cloud-agent-e2e-lib.mjs';

const MODEL = process.env.ASKPDF_E2E_LLM_MODEL || 'inclusionai/ling-3.2-flash:free';
const PDF = process.env.ASKPDF_E2E_PDF || path.join(REPO_ROOT, 'rag_service/tests/01030000000000.pdf');
const QUESTION = process.env.ASKPDF_CLARIFY_QUESTION || 'Compare them.';
const CUSTOM = process.env.ASKPDF_CLARIFY_CUSTOM || 'Summarize the architecture section in one paragraph.';

function md5File(filePath) {
  const data = fs.readFileSync(filePath);
  return crypto.createHash('md5').update(data).digest('hex');
}

async function seedComposerFromOption(page, composer) {
  const decisionPanel = clarificationPanel(page);
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
  await decisionPanel.locator('button[aria-label^="Send "]').nth(optionIndex).click();
  await composer.waitFor({ state: 'visible', timeout: 30_000 });
  const seeded = (await composer.inputValue()).trim();
  if (!seeded.includes(optionText.slice(0, Math.min(24, optionText.length)))) {
    throw new Error(`Composer was not seeded from option; expected fragment of ${JSON.stringify(optionText)}`);
  }
  return optionText;
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

  const composer = page.getByTestId('chat-composer-input');

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

  const panel = clarificationPanel(page);
  await guard.capture(page, 'clarification-06-card-before-cancel.png', { requireChatWorkspace: true });
  await panel.getByRole('button', { name: /Close decision panel/i }).evaluate((node) => node.click());
  await waitForComposerReady(page);
  const reseed = (await composer.inputValue()).trim();
  if (!reseed.includes(QUESTION)) {
    throw new Error(`Cancel did not re-seed composer with ${JSON.stringify(QUESTION)}; got ${JSON.stringify(reseed)}`);
  }
  await guard.capture(page, 'clarification-07-cancel-reseed.png', { requireChatWorkspace: true });

  await chatRateLimitPause();
  await typeAndSendChat(page, QUESTION);
  await waitClarificationCard(page);
  await guard.capture(page, 'clarification-08-card-after-resend.png', { requireChatWorkspace: true });

  await page.getByRole('button', { name: /Something else/i }).click();
  await page.getByRole('textbox', { name: 'Your response' }).fill(CUSTOM);
  await guard.capture(page, 'clarification-09-custom-filled.png', { requireChatWorkspace: true });
  await page.getByRole('button', { name: 'Send custom response' }).click();
  await composer.waitFor({ state: 'visible', timeout: 30_000 });
  const customSeeded = (await composer.inputValue()).trim();
  if (customSeeded !== CUSTOM.trim()) {
    throw new Error(`Custom response did not seed composer; expected ${JSON.stringify(CUSTOM)}; got ${JSON.stringify(customSeeded)}`);
  }
  await guard.capture(page, 'clarification-10-custom-composer-seeded.png', { requireChatWorkspace: true });

  await chatRateLimitPause();
  await typeAndSendChat(page, customSeeded);
  await waitOutOfSending(page);
  await waitForComposerReady(page);
  let customAnswer = await chatTranscriptText(page);
  assertGroundedPdfAnswer(customAnswer, { extraPattern: /(architecture|MORAN|encoder|layer|network)/i });
  await guard.capture(page, 'clarification-11-custom-grounded-answer.png', { requireChatWorkspace: true });

  await chatRateLimitPause();
  await typeAndSendChat(page, QUESTION);
  await waitClarificationCard(page);
  await guard.capture(page, 'clarification-12-card-for-option-flow.png', { requireChatWorkspace: true });

  const optionText = await seedComposerFromOption(page, composer);
  await guard.capture(page, 'clarification-13-composer-seeded.png', { requireChatWorkspace: true });

  const edited = `${(await composer.inputValue()).trim().replace(/\s+$/, '')} (edited for send)`;
  await composer.fill(edited);
  await guard.capture(page, 'clarification-14-composer-edited.png', { requireChatWorkspace: true });

  await chatRateLimitPause();
  await typeAndSendChat(page, edited);
  await waitOutOfSending(page);
  await waitForComposerReady(page);
  const optionAnswer = await chatTranscriptText(page);
  assertGroundedPdfAnswer(optionAnswer);
  await guard.capture(page, 'clarification-15-grounded-answer.png', { requireChatWorkspace: true });

  await chatRateLimitPause();
  await waitForComposerReady(page);
  await typeAndSendChat(page, QUESTION);
  const repeatOutcome = await waitRepeatClarifyOrGrounded(page);
  if (repeatOutcome === 'clarification') {
    await guard.capture(page, 'clarification-16-repeat-compare-card.png', { requireChatWorkspace: true });
    await clarificationPanel(page).getByRole('button', { name: /Close decision panel/i }).evaluate((node) => node.click());
  } else {
    await waitOutOfSending(page);
    await waitForComposerReady(page);
    const repeatText = await chatTranscriptText(page);
    assertNotScopeDecline(repeatText);
    assertGroundedPdfAnswer(repeatText);
    await guard.capture(page, 'clarification-16-repeat-compare-answer.png', { requireChatWorkspace: true });
  }

  const screenshotMd5 = Object.fromEntries(
    guard.paths.map((p) => [path.basename(p), md5File(p)]),
  );

  const meta = {
    tested_sha: sha,
    model: MODEL,
    question: QUESTION,
    picked_option: optionText,
    sent_question: edited,
    custom_question: CUSTOM,
    repeat_compare_outcome: repeatOutcome,
    assistant_answer_preview: optionAnswer.slice(0, 280),
    custom_answer_preview: customAnswer.slice(0, 280),
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