import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { resolveE2eLlmModel } from './e2e-llm-default.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ART = process.env.ASKPDF_E2E_ARTIFACTS || '/opt/cursor/artifacts';
const BASE = process.env.ASKPDF_E2E_BASE_URL || 'http://127.0.0.1:3000';
const MODEL = resolveE2eLlmModel();

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

async function openChatSurface(page) {
  const threadTab = page.getByRole('tab', { name: /^Thread$/i });
  if (await threadTab.isVisible({ timeout: 3000 }).catch(() => false)) {
    await threadTab.click();
  }
  await focusChatComposer(page).catch(() => undefined);
}

async function ensureChatWorkspace(page) {
  await page.goto(BASE, { waitUntil: 'domcontentloaded', timeout: 120000 });
  await page.waitForTimeout(1500);
  const llmCombo = page.getByRole('combobox', { name: /Select LLM/i }).first();
  const composerReady = page.getByPlaceholder(/Ask a question about your documents/i).first();
  if (
    await llmCombo.isVisible({ timeout: 3000 }).catch(() => false)
    && await composerReady.isVisible({ timeout: 1000 }).catch(() => false)
  ) {
    await openChatSurface(page);
    return;
  }
  let createInExisting = page.getByRole('button', { name: /Create thread in web-approval-test/i }).first();
  if (!(await createInExisting.isVisible({ timeout: 12000 }).catch(() => false))) {
    await page.getByRole('button', { name: /Create Project/i }).click();
    const newLabel = `web-approval-${Date.now()}`;
    await page.getByRole('textbox').first().fill(newLabel);
    await page.getByRole('button', { name: /^Create Project$/i }).click();
    await page.waitForTimeout(1200);
    createInExisting = page
      .getByRole('button', { name: new RegExp(`Create thread in ${newLabel.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`, 'i') })
      .first();
  }
  await createInExisting.scrollIntoViewIfNeeded();
  await createInExisting.click({ force: true });
  const threadName = `approval-live-${Date.now()}`;
  const nameField = page.getByRole('dialog').getByRole('textbox').first();
  await nameField.waitFor({ state: 'visible', timeout: 30000 });
  await nameField.fill(threadName);
  await page.getByRole('dialog').getByRole('button', { name: /^Create$/i }).click();
  await llmCombo.waitFor({ state: 'visible', timeout: 120000 });
  await openChatSurface(page);
}

async function createFreshThread(page) {
  const createThread = page.getByRole('button', { name: /Create Thread/i }).first();
  if (await createThread.isVisible({ timeout: 3000 }).catch(() => false)) {
    await createThread.click();
    await page.getByRole('textbox').first().fill(`approval-live-${Date.now()}`);
    await page.getByRole('button', { name: /^Create$/i }).click();
    await page.getByRole('combobox', { name: /Select LLM/i }).first().waitFor({ timeout: 60000 });
  }
}

async function focusChatComposer(page) {
  const composer = page.getByPlaceholder(/Ask a question about your documents/i).first();
  await composer.scrollIntoViewIfNeeded();
  return composer;
}

async function waitForComposerReady(page) {
  const checking = page.getByPlaceholder(/Checking LLM model/i);
  if (await checking.isVisible({ timeout: 2000 }).catch(() => false)) {
    await checking.waitFor({ state: 'hidden', timeout: 120000 });
  }
  const composer = await focusChatComposer(page);
  await composer.waitFor({ state: 'visible', timeout: 120000 });
}

async function setWebSearchAsk(page) {
  const labels = ['Internet Search Off', 'Internet Search On', 'Ask me every time before internet search'];
  const btn = page.getByRole('button', { name: new RegExp(labels.join('|')) }).first();
  await btn.scrollIntoViewIfNeeded();
  for (let i = 0; i < labels.length; i += 1) {
    const aria = await btn.getAttribute('aria-label');
    if (aria === labels[2]) return;
    await btn.click();
    await page.waitForTimeout(350);
  }
}

async function pickModel(page, model) {
  const combo = page.getByRole('combobox', { name: /Select LLM/i }).first();
  await combo.waitFor({ state: 'visible', timeout: 120000 });
  await combo.scrollIntoViewIfNeeded();
  const needle = model.includes('/') ? model.split('/').slice(-1)[0] : model;
  await combo.click();
  await combo.fill(needle);
  await page.waitForTimeout(800);
  const option = page.getByRole('option', { name: new RegExp(needle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i') }).first();
  await option.waitFor({ state: 'visible', timeout: 30000 });
  await option.click();
  await waitForComposerReady(page);
}

async function dismissClarificationIfPresent(page) {
  const prompt = page.getByText(/Did you mean one of these/i);
  if (!(await prompt.isVisible({ timeout: 1500 }).catch(() => false))) return;
  const original = page.getByRole('button', { name: /Original question/i }).first();
  if (await original.isVisible({ timeout: 3000 }).catch(() => false)) {
    await original.click();
    return;
  }
  await page.getByRole('button', { name: /Close decision panel/i }).click({ timeout: 3000 }).catch(() => undefined);
}

async function sendChat(page, text) {
  const input = page.getByPlaceholder(/Ask a question about your documents/i).first();
  await input.fill(text);
  await page.keyboard.press('Enter');
  await dismissClarificationIfPresent(page);
}

async function waitForComposerIdle(page) {
  await expect(page.getByPlaceholder(/Ask a question about your documents/i).first()).toBeEnabled({ timeout: 240000 });
}

async function assertApprovalCard(page) {
  await expect(page.getByText(/auto_rag_agent/i)).toHaveCount(0);
  const title = page.getByText(/Allow a web search\?/i).first();
  await title.scrollIntoViewIfNeeded();
  await expect(title).toBeVisible({ timeout: 180000 });
  const approveOnce = page.getByRole('button', { name: /Approve once/i });
  const skip = page.getByRole('button', { name: /Skip this tool for this/i });
  await approveOnce.scrollIntoViewIfNeeded();
  await expect(approveOnce).toBeVisible();
  await expect(skip).toBeVisible();
  const args = page.locator('pre').filter({ hasText: /query/i });
  if (await args.count()) {
    await expect(args.first()).toBeVisible();
  } else {
    await expect(page.getByText(/"query"/i).first()).toBeVisible();
  }
  const vp = page.viewportSize();
  for (const loc of [approveOnce, skip]) {
    const box = await loc.boundingBox();
    expect(box).not.toBeNull();
    expect(box.y + box.height).toBeLessThanOrEqual((vp?.height || 900) + 2);
  }
}

async function assertViewportScreenshot(page, filePath) {
  const vp = page.viewportSize();
  expect(vp?.width).toBeGreaterThan(0);
  const inner = await page.evaluate(() => ({ w: window.innerWidth, h: window.innerHeight }));
  expect(inner.w).toBe(vp?.width);
  await page.screenshot({ path: filePath, fullPage: false });
  const buf = fs.readFileSync(filePath);
  const pngWidth = buf.readUInt32BE(16);
  expect(pngWidth).toBe(vp?.width);
}

async function waitForAnswer(page, pattern) {
  await waitForComposerIdle(page);
  const answer = page.locator('li, .MuiListItem-root').filter({ hasText: pattern }).filter({
    hasNotText: /Use the internet search tool to find|What is the weather in Paris today/i,
  }).last();
  await answer.waitFor({ state: 'visible', timeout: 300000 });
  const text = await answer.innerText();
  expect(text.length).toBeGreaterThan(15);
  expect(text).not.toMatch(/Approved, continuing|technical error while processing/i);
  await expect(page.getByText(/auto_rag_agent/i)).toHaveCount(0);
  return text;
}

async function prepareThread(page) {
  await ensureChatWorkspace(page);
  await openChatSurface(page);
  await pickModel(page, MODEL);
  await selectRouterAgentWorkflow(page);
  await pickModel(page, MODEL);
  await setWebSearchAsk(page);
  await expect(page.getByText(/auto_rag_agent/i)).toHaveCount(0);
  await waitForComposerReady(page);
}

async function runApproveFlow(page, suffix) {
  await prepareThread(page);
  await sendChat(page, 'Use the internet search tool to find the current USD to EUR exchange rate. Reply with one number.');
  await assertApprovalCard(page);
  await assertViewportScreenshot(page, path.join(ART, `chat-tool-approval-${suffix}-prompt.png`));
  await page.getByRole('button', { name: /Approve once/i }).click({ force: true });
  await waitForAnswer(page, /EUR|USD|exchange|dollar|euro|Research canvas|Web sources|0\.\d/i);
  await assertViewportScreenshot(page, path.join(ART, `chat-tool-approval-${suffix}-answer.png`));
}

async function runSkipFlow(page, suffix) {
  await prepareThread(page);
  await sendChat(page, 'What is the weather in Paris today? Reply in one sentence.');
  await assertApprovalCard(page);
  await assertViewportScreenshot(page, path.join(ART, `chat-tool-approval-${suffix}-skip-prompt.png`));
  await page.getByRole('button', { name: /Skip this tool for this/i }).click({ force: true });
  await waitForAnswer(page, /Paris|weather|without|unable|sorry|skip|forecast/i);
  await assertViewportScreenshot(page, path.join(ART, `chat-tool-approval-${suffix}-skip-answer.png`));
}

const viewports = [
  { suffix: '1440x900', width: 1440, height: 900 },
  { suffix: '1024x768', width: 1024, height: 768 },
];

test.describe('Live chat web search approval UI', () => {
  test.describe.configure({ timeout: 900_000 });
  test.beforeAll(() => fs.mkdirSync(ART, { recursive: true }));

  for (const vp of viewports) {
    test(`approve @ ${vp.suffix}`, async ({ browser }) => {
      const context = await browser.newContext({ viewport: { width: vp.width, height: vp.height } });
      const page = await context.newPage();
      try {
        await runApproveFlow(page, vp.suffix);
      } finally {
        await context.close();
      }
    });

    test(`skip @ ${vp.suffix}`, async ({ browser }) => {
      const context = await browser.newContext({ viewport: { width: vp.width, height: vp.height } });
      const page = await context.newPage();
      try {
        await runSkipFlow(page, vp.suffix);
      } finally {
        await context.close();
      }
    });
  }
});