#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';
import sharp from 'sharp';

const BASE = process.env.ASKPDF_E2E_BASE_URL || 'http://127.0.0.1:3000';
const API_BASE = process.env.ASKPDF_E2E_API_URL || 'http://127.0.0.1:8000';
const ART = process.env.ASKPDF_E2E_ARTIFACTS || '/opt/cursor/artifacts';
const MODEL = process.env.ASKPDF_E2E_LLM_MODEL || 'inclusionai/ling-3.2-flash:free';
const VP = { width: 1440, height: 900, name: '1440x900' };

fs.mkdirSync(ART, { recursive: true });

async function verifyShot(filePath) {
  const meta = await sharp(filePath).metadata();
  if (meta.width !== VP.width || meta.height !== VP.height) {
    throw new Error(`${filePath}: expected ${VP.width}x${VP.height}, got ${meta.width}x${meta.height}`);
  }
  console.log(`verified ${filePath} ${meta.width}x${meta.height}`);
}

async function shot(page, file) {
  const filePath = path.join(ART, file);
  await page.waitForTimeout(800);
  await page.screenshot({ path: filePath, fullPage: false });
  await verifyShot(filePath);
}

async function ensureThread(page) {
  await page.goto(BASE, { waitUntil: 'domcontentloaded', timeout: 120000 });
  await page.waitForTimeout(1500);
  const llm = page.getByRole('combobox', { name: /Select LLM/i }).first();
  if (await llm.isVisible({ timeout: 3000 }).catch(() => false)) {
    return;
  }
  const createProject = page.getByRole('button', { name: /Create Project/i });
  if (await createProject.isVisible({ timeout: 5000 }).catch(() => false)) {
    await createProject.click();
    await page.getByRole('textbox').first().fill(`browser-e2e-${Date.now()}`);
    await page.getByRole('button', { name: /^Create Project$/i }).click();
    await page.waitForTimeout(1000);
  }
  const createThread = page.getByRole('button', { name: /Create Thread/i }).first();
  await createThread.waitFor({ timeout: 60000 });
  await createThread.click();
  await page.getByRole('textbox').first().fill(`browser-flow-${Date.now()}`);
  const [createResponse] = await Promise.all([
    page.waitForResponse(
      (response) => response.url().includes('/api/threads') && response.request().method() === 'POST',
      { timeout: 60000 },
    ),
    page.getByRole('button', { name: /^Create$/i }).click(),
  ]);
  const created = await createResponse.json();
  if (created?.id) cachedThreadId = created.id;
  await llm.waitFor({ timeout: 60000 });
}

async function selectRouterWorkflow(page) {
  const settings = page.getByRole('button', { name: /AI prompt settings for this thread/i }).first();
  await settings.click();
  const dialog = page.getByRole('dialog');
  await dialog.waitFor({ timeout: 15000 });
  const workflow = dialog.getByRole('combobox', { name: /^Agent workflow/i });
  await workflow.click();
  const router = page.getByRole('option', { name: /Router Agent/i }).first();
  await router.waitFor({ timeout: 15000 });
  await router.click();
  await dialog.getByRole('button', { name: /^Save$/i }).click();
  await dialog.waitFor({ state: 'hidden', timeout: 15000 });
  await page.waitForTimeout(500);
}

async function pickModel(page) {
  const combo = page.getByRole('combobox', { name: /Select LLM/i }).first();
  await combo.click();
  const token = MODEL.split('/').pop() || MODEL;
  await combo.fill(token.replace(/:free$/, ''));
  await page.waitForTimeout(600);
  const escaped = MODEL.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const option = page.getByRole('option', { name: new RegExp(escaped, 'i') }).first();
  if (await option.count()) {
    await option.click();
  } else {
    await page.keyboard.press('Escape');
  }
  await page.waitForTimeout(400);
}

async function setWebSearchOff(page) {
  const offBtn = page.getByRole('button', { name: /^Internet Search Off$/i });
  if (await offBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
    return;
  }
  const toggle = page
    .getByRole('button', {
      name: /Internet Search Off|Internet Search On|Ask me every time before internet search/i,
    })
    .first();
  for (let i = 0; i < 5; i += 1) {
    await toggle.click();
    await page.waitForTimeout(300);
    if (await offBtn.isVisible({ timeout: 500 }).catch(() => false)) {
      return;
    }
  }
}

async function setBrowserAskMode(page) {
  await page.evaluate(() => {
    localStorage.setItem('last_browser_access_mode', 'ask');
    window.dispatchEvent(new CustomEvent('askpdf:browser-access-mode', { detail: 'ask' }));
  });
  const askToggle = page.getByRole('button', { name: /^Ask before the agent uses the workbench browser$/i });
  if (await askToggle.isVisible({ timeout: 3000 }).catch(() => false)) {
    return;
  }
  const browserToggle = page
    .getByRole('button', {
      name: /Browser agent off|Browser agent on|Ask before the agent uses the workbench browser/i,
    })
    .first();
  for (let i = 0; i < 5; i += 1) {
    await browserToggle.click();
    await page.waitForTimeout(350);
    if (await askToggle.isVisible({ timeout: 1000 }).catch(() => false)) {
      return;
    }
  }
  throw new Error('Could not set browser access mode to Ask');
}

async function openBrowserTab(page) {
  const browserTab = page.getByRole('tab', { name: /^Browser$/i }).first();
  await browserTab.waitFor({ timeout: 30000 });
  await browserTab.click();
  await page.waitForTimeout(2500);
}

async function openConversation(page) {
  const threadTab = page.getByRole('tab', { name: /^Thread$/i }).first();
  await threadTab.waitFor({ timeout: 30000 });
  await threadTab.click();
  await page.waitForTimeout(500);
}

let cachedThreadId = null;

async function resolveThreadId(page) {
  if (cachedThreadId) return cachedThreadId;
  const fromUrl = await page.evaluate(() => {
    const m = window.location.pathname.match(/threads\/([^/]+)/);
    return m ? m[1] : null;
  });
  if (fromUrl) {
    cachedThreadId = fromUrl;
    return fromUrl;
  }
  throw new Error('thread id not found');
}

async function sendChat(page, text) {
  const composer = page.getByPlaceholder(/Ask a question about your documents/i).first();
  await composer.waitFor({ timeout: 60000 });
  await composer.fill(text);
  await page.getByRole('button', { name: /^Send$/i }).first().click();
}

async function waitForApproval(page, timeoutMs = 180000) {
  const approve = page.getByRole('button', { name: /^Approve once$/i }).first();
  await approve.waitFor({ timeout: timeoutMs });
  return approve;
}

const browser = await chromium.launch({
  headless: process.env.ASKPDF_E2E_HEADLESS !== '0',
  channel: process.env.ASKPDF_E2E_HEADLESS === '0' ? 'chrome' : undefined,
});
const context = await browser.newContext({ viewport: { width: VP.width, height: VP.height } });
const page = await context.newPage();

await ensureThread(page);
await pickModel(page);
await selectRouterWorkflow(page);
await openConversation(page);
await setWebSearchOff(page);
await setBrowserAskMode(page);
await openBrowserTab(page);
if (process.env.ASKPDF_E2E_HEADLESS === '0') {
  await page.waitForTimeout(12000);
}
await shot(page, 'e2e-01-browser-workbench-tab-1440x900.png');

await shot(page, 'e2e-02-browser-access-ask-mode-1440x900.png');

const prompt =
  'Use the shared workbench browser: open https://example.com and answer in one sentence what the main heading on the page says (the visible h1 text). I have no uploaded documents about this page.';
await sendChat(page, prompt);

try {
  var approve = await waitForApproval(page);
} catch (error) {
  await shot(page, 'e2e-failure-before-approval-1440x900.png');
  throw error;
}
await shot(page, 'e2e-03-tool-approval-card-1440x900.png');
await approve.click();
await page
  .locator('div.MuiPaper-root')
  .filter({ hasText: /Example Domain/i })
  .last()
  .waitFor({ state: 'visible', timeout: 300000 });
await page.waitForTimeout(2000);
await shot(page, 'e2e-04-agent-answer-from-page-1440x900.png');

await sendChat(
  page,
  'Save the current workbench browser page to this thread as a document, then confirm the file name.',
);
try {
  const saveApprove = await waitForApproval(page, 120000);
  await saveApprove.click();
} catch {
  /* browser_save_to_thread may be on without a second approval card */
}
await page.waitForTimeout(25000);
const docsTab = page.getByRole('tab', { name: /^Documents$/i }).first();
await docsTab.click();
await page.waitForTimeout(2500);
await page
  .getByRole('button', { name: /Example Domain|example\.com/i })
  .first()
  .waitFor({ timeout: 180000 });
await shot(page, 'e2e-05-page-saved-to-thread-1440x900.png');

await context.close();
await browser.close();
console.log('browser-flow-e2e complete');