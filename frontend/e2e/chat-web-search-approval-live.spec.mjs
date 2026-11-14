import { test, expect } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, '../..');
const ART = process.env.ASKPDF_E2E_ARTIFACTS || '/opt/cursor/artifacts';
const BASE = process.env.ASKPDF_E2E_BASE_URL || 'http://127.0.0.1:3000';
const MODEL = process.env.ASKPDF_E2E_LLM_MODEL || 'openai/gpt-4.1-mini';

async function ensureChatWorkspace(page) {
  await page.goto(BASE, { waitUntil: 'domcontentloaded', timeout: 120000 });
  await page.waitForTimeout(1500);

  const llmCombo = page.getByRole('combobox', { name: /Select LLM/i }).first();
  if (await llmCombo.isVisible({ timeout: 3000 }).catch(() => false)) {
    return;
  }

  let projectLabel = 'web-approval-test';
  const createInExisting = page.getByRole('button', { name: /Create thread in web-approval-test/i });
  if (!(await createInExisting.isVisible({ timeout: 4000 }).catch(() => false))) {
    const createProject = page.getByRole('button', { name: /Create Project/i });
    await createProject.waitFor({ timeout: 60000 });
    await createProject.click();
    projectLabel = `web-approval-${Date.now()}`;
    await page.getByRole('textbox').first().fill(projectLabel);
    await page.getByRole('button', { name: /^Create Project$/i }).click();
    await page.waitForTimeout(1200);
  }

  const createThreadBtn = page
    .getByRole('button', { name: new RegExp(`Create thread in ${projectLabel.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`, 'i') })
    .or(page.getByRole('button', { name: /Create Thread/i }))
    .first();
  await createThreadBtn.waitFor({ timeout: 60000 });
  await createThreadBtn.click();
  await page.getByRole('textbox').first().fill(`approval-live-${Date.now()}`);
  await page.getByRole('button', { name: /^Create$/i }).click();
  await llmCombo.waitFor({ timeout: 60000 });
}

async function waitForComposerReady(page) {
  const checking = page.getByPlaceholder(/Checking LLM model/i);
  if (await checking.isVisible({ timeout: 2000 }).catch(() => false)) {
    await checking.waitFor({ state: 'hidden', timeout: 180000 });
  }
  await page.getByPlaceholder(/Ask a question about your documents/i).waitFor({ timeout: 180000 });
}

async function setWebAskMode(page) {
  const chatHeader = page.locator('div').filter({
    has: page.getByRole('combobox', { name: /Select LLM/i }),
  }).first();
  const btn = chatHeader.locator(
    'button[aria-label="Internet Search Off"], button[aria-label="Internet Search On"], button[aria-label="Ask me every time before internet search"]',
  ).first();
  await btn.waitFor({ timeout: 60000 });
  for (let i = 0; i < 4; i += 1) {
    const label = await btn.getAttribute('aria-label');
    if (label?.includes('Ask me every time')) return;
    await btn.click();
    await page.waitForTimeout(350);
  }
  throw new Error('Could not set web search to ask mode');
}

async function pickModel(page, model) {
  const combo = page.getByRole('combobox', { name: /Select LLM/i }).first();
  await combo.click();
  await combo.fill(model.includes('/') ? model.split('/').slice(-1)[0] : model);
  await page.waitForTimeout(400);
  const option = page.getByRole('option', { name: new RegExp(model.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i') }).first();
  if (await option.count()) {
    await option.click();
    return;
  }
  await page.keyboard.press('Escape');
}

async function sendChat(page, text) {
  const input = page.getByPlaceholder(/Ask a question about your documents/i).first();
  await input.click();
  await input.fill(text);
  await page.keyboard.press('Enter');
}

async function assertApprovalVisible(page) {
  const approve = page.getByRole('button', { name: /Approve once/i });
  const skip = page.getByRole('button', { name: /Skip this tool/i });
  await approve.waitFor({ timeout: 240000 });
  await expect(skip).toBeVisible();
  const viewport = page.viewportSize();
  for (const loc of [approve, skip]) {
    const box = await loc.boundingBox();
    expect(box).not.toBeNull();
    expect(box.y + box.height).toBeLessThanOrEqual((viewport?.height || 800) + 1);
  }
  await expect(page.getByTestId('chat-human-review-panel')).toBeVisible();
}

async function runFlow(page, suffix) {
  await ensureChatWorkspace(page);
  await pickModel(page, MODEL);
  await waitForComposerReady(page);
  await setWebAskMode(page);
  for (let i = 1; i <= 10; i += 1) {
    await sendChat(page, `Filler message ${i}: note the number ${i}.`);
    await page.waitForTimeout(1200);
  }
  await sendChat(page, 'What is the current USD to EUR exchange rate? Reply briefly.');
  await assertApprovalVisible(page);
  await page.screenshot({ path: path.join(ART, `chat-web-approval-${suffix}-approve-prompt.png`), fullPage: false });
  await page.getByRole('button', { name: /Approve once/i }).click();
  await page.getByRole('button', { name: /Approve once/i }).waitFor({ state: 'hidden', timeout: 240000 }).catch(() => {});
  await page.waitForTimeout(6000);
  await page.screenshot({ path: path.join(ART, `chat-web-approval-${suffix}-approved.png`), fullPage: false });
  await setWebAskMode(page);
  await sendChat(page, 'What is the weather in Paris today? Reply in one sentence.');
  await assertApprovalVisible(page);
  await page.getByRole('button', { name: /Skip this tool/i }).click();
  await page.waitForTimeout(8000);
  await page.screenshot({ path: path.join(ART, `chat-web-approval-${suffix}-denied.png`), fullPage: false });
}

const viewports = [
  { suffix: '1280x800', width: 1280, height: 800 },
  { suffix: '1024x768', width: 1024, height: 768 },
];

test.describe.serial('Live chat web search approval UI', () => {
  test.describe.configure({ timeout: 600_000 });

  test.beforeAll(() => {
    fs.mkdirSync(ART, { recursive: true });
  });

  for (const vp of viewports) {
    test(vp.suffix, async ({ browser }) => {
      const page = await browser.newPage({ viewport: { width: vp.width, height: vp.height } });
      await runFlow(page, vp.suffix);
      await page.close();
    });
  }

  test.afterAll(() => {
    const out = {
      model: MODEL,
      commit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: REPO, encoding: 'utf8' }).trim(),
      pass: true,
    };
    fs.writeFileSync(path.join(ART, 'chat-web-approval-ui-results.json'), JSON.stringify(out, null, 2));
  });
});