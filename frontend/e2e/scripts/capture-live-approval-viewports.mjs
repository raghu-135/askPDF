/**
 * Live capture for 1440×900 and 1024×768 viewports (real stack; not CI).
 * Shared setup with chat-web-search-approval-live.spec.mjs.
 */
import { chromium } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { resolveE2eLlmModel } from '../e2e-llm-default.mjs';

const ART = process.env.ASKPDF_E2E_ARTIFACTS || '/opt/cursor/artifacts';
const BASE = process.env.ASKPDF_E2E_BASE_URL || 'http://127.0.0.1:3000';
const MODEL = resolveE2eLlmModel();

async function createThread(page) {
  await page.goto(BASE, { waitUntil: 'domcontentloaded', timeout: 120000 });
  await page.waitForTimeout(1200);
  let createBtn = page.getByRole('button', { name: /Create thread in web-approval-test/i }).first();
  if (!(await createBtn.isVisible({ timeout: 12000 }).catch(() => false))) {
    await page.getByRole('button', { name: /Create Project/i }).click();
    const label = `web-approval-${Date.now()}`;
    await page.getByRole('textbox').first().fill(label);
    await page.getByRole('button', { name: /^Create Project$/i }).click();
    createBtn = page.getByRole('button', { name: new RegExp(`Create thread in ${label}`, 'i') }).first();
  }
  await createBtn.scrollIntoViewIfNeeded();
  await createBtn.click({ force: true });
  await page.getByRole('dialog').getByRole('textbox').first().fill(`proof-${Date.now()}`);
  await page.getByRole('dialog').getByRole('button', { name: /^Create$/i }).click();
  await page.getByRole('combobox', { name: /Select LLM/i }).first().waitFor({ timeout: 120000 });
}

async function pickModel(page) {
  const combo = page.getByRole('combobox', { name: /Select LLM/i }).first();
  const needle = MODEL.split('/').slice(-1)[0];
  await combo.scrollIntoViewIfNeeded();
  await combo.click();
  await combo.fill(needle);
  await page.waitForTimeout(800);
  await page.getByRole('option', { name: new RegExp(needle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i') }).first().click();
  await page.getByPlaceholder(/Ask a question about your documents/i).first().waitFor({ timeout: 120000 });
}

async function selectRouter(page) {
  await page.getByRole('button', { name: /AI prompt settings for this thread/i }).click();
  const dialog = page.getByRole('dialog');
  await dialog.waitFor();
  await dialog.getByRole('combobox', { name: /Agent workflow/i }).click();
  await page.getByRole('option', { name: /Router Agent/i }).click();
  await dialog.getByRole('button', { name: /^Save$/i }).click();
  await dialog.waitFor({ state: 'hidden', timeout: 60000 });
  await pickModel(page);
}

async function setAskMode(page) {
  const labels = ['Internet Search Off', 'Internet Search On', 'Ask me every time before internet search'];
  const btn = page.getByRole('button', { name: new RegExp(labels.join('|')) }).first();
  await btn.scrollIntoViewIfNeeded();
  for (let i = 0; i < labels.length; i += 1) {
    if ((await btn.getAttribute('aria-label')) === labels[2]) return;
    await btn.click({ force: true });
    await page.waitForTimeout(350);
  }
}

async function dismissClarify(page) {
  const original = page.getByRole('button', { name: /Original question/i }).first();
  if (await original.isVisible({ timeout: 2000 }).catch(() => false)) await original.click();
}

async function waitComposerIdle(page) {
  const sending = page.getByPlaceholder(/Sending\.\.\.|Checking LLM model/i);
  if (await sending.isVisible({ timeout: 2000 }).catch(() => false)) {
    await sending.waitFor({ state: 'hidden', timeout: 300000 });
  }
  await page.getByPlaceholder(/Ask a question about your documents/i).waitFor({ timeout: 300000 });
}

async function pollForAnswer(page, predicate) {
  for (let i = 0; i < 120; i += 1) {
    await waitComposerIdle(page);
    const approvalOpen = await page.getByText(/Allow a web search\?/i).isVisible().catch(() => false);
    const messages = await page.locator('li, .MuiListItem-root').allInnerTexts();
    const hit = messages.find((t) => predicate(t));
    if (hit && !approvalOpen) return hit;
    await page.waitForTimeout(3000);
  }
  throw new Error('Timed out waiting for assistant answer');
}

function assertPngWidth(filePath, width) {
  const w = fs.readFileSync(filePath).readUInt32BE(16);
  if (w !== width) throw new Error(`PNG width ${w} != ${width} for ${filePath}`);
}

async function captureViewport(vp) {
  const suffix = `${vp.width}x${vp.height}`;
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: vp });
  try {
    await createThread(page);
    await pickModel(page);
    await selectRouter(page);
    await setAskMode(page);
    const input = page.getByPlaceholder(/Ask a question about your documents/i).first();
    await input.fill('Use the internet search tool to find the current USD to EUR exchange rate. Reply with one number.');
    await page.keyboard.press('Enter');
    await dismissClarify(page);
    const title = page.getByText(/Allow a web search\?/i).first();
    const approve = page.getByRole('button', { name: /Approve once/i });
    const skip = page.getByRole('button', { name: /Skip this tool for this/i });
    await title.waitFor({ state: 'visible', timeout: 180000 });
    await title.scrollIntoViewIfNeeded();
    await approve.scrollIntoViewIfNeeded();
    await skip.scrollIntoViewIfNeeded();
    const promptPath = path.join(ART, `chat-tool-approval-${suffix}-prompt.png`);
    await page.screenshot({ path: promptPath, fullPage: false });
    assertPngWidth(promptPath, vp.width);
    for (let attempt = 0; attempt < 4; attempt += 1) {
      const approvalOpen = await title.isVisible().catch(() => false);
      if (approvalOpen) {
        await approve.click({ force: true });
      }
      try {
        await pollForAnswer(
          page,
          (t) => (
            (/0\.\d{2,}/.test(t) || /\b\d+\.\d+\b/.test(t) || /EUR|USD|exchange/i.test(t))
            && !/APPROVING|Approve search_web|Allow a web search/i.test(t)
            && t.length > 30
          ),
        );
        break;
      } catch (err) {
        if (attempt === 3) throw err;
      }
    }
    const answerPath = path.join(ART, `chat-tool-approval-${suffix}-answer.png`);
    await page.screenshot({ path: answerPath, fullPage: false });
    assertPngWidth(answerPath, vp.width);
    await input.fill('What is the weather in Paris today? Reply in one sentence.');
    await page.keyboard.press('Enter');
    await dismissClarify(page);
    await title.waitFor({ state: 'visible', timeout: 180000 });
    await skip.scrollIntoViewIfNeeded();
    const skipPromptPath = path.join(ART, `chat-tool-approval-${suffix}-skip-prompt.png`);
    await page.screenshot({ path: skipPromptPath, fullPage: false });
    assertPngWidth(skipPromptPath, vp.width);
    await skip.click({ force: true });
    await pollForAnswer(
      page,
      (t) => /Paris|weather|without|unable|skip|cannot/i.test(t)
        && !/Approve search_web|Allow a web search/i.test(t)
        && t.length > 40,
    );
    const skipAnswerPath = path.join(ART, `chat-tool-approval-${suffix}-skip-answer.png`);
    await page.screenshot({ path: skipAnswerPath, fullPage: false });
    assertPngWidth(skipAnswerPath, vp.width);
    console.log('OK', suffix);
  } finally {
    await browser.close();
  }
}

fs.mkdirSync(ART, { recursive: true });
const viewports = process.env.ASKPDF_E2E_VIEWPORTS
  ? process.env.ASKPDF_E2E_VIEWPORTS.split(',').map((pair) => {
    const [width, height] = pair.split('x').map(Number);
    return { width, height };
  })
  : [{ width: 1440, height: 900 }, { width: 1024, height: 768 }];
for (const vp of viewports) {
  await captureViewport(vp);
}