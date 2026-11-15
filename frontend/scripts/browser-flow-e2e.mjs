#!/usr/bin/env node
import crypto from 'node:crypto';
import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';
import sharp from 'sharp';

const BASE = process.env.ASKPDF_E2E_BASE_URL || 'http://127.0.0.1:3000';
const ART = process.env.ASKPDF_E2E_ARTIFACTS || '/opt/cursor/artifacts';
const MODEL = process.env.ASKPDF_E2E_LLM_MODEL || 'inclusionai/ling-3.2-flash:free';
const VP = { width: 1440, height: 900, name: '1440x900' };

fs.mkdirSync(ART, { recursive: true });

const shotMd5s = new Map();

async function waitForCapturePage(fragment, timeoutMs = 240000) {
  const deadline = Date.now() + timeoutMs;
  const pattern = new RegExp(fragment, 'i');
  while (Date.now() < deadline) {
    try {
      const raw = execSync(
        'docker exec askpdf-browser-capture-1 curl -sf http://127.0.0.1:8080/read',
        { encoding: 'utf8', timeout: 20000 },
      );
      const data = JSON.parse(raw);
      const hay = `${data.title || ''}\n${data.url || ''}\n${data.text || ''}`;
      if (pattern.test(hay)) {
        return data;
      }
    } catch {
      /* capture API may be busy until navigation completes */
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  throw new Error(`Timed out waiting for capture page matching /${fragment}/i`);
}

async function verifyShot(filePath) {
  const meta = await sharp(filePath).metadata();
  if (meta.width !== VP.width || meta.height !== VP.height) {
    throw new Error(`${filePath}: expected ${VP.width}x${VP.height}, got ${meta.width}x${meta.height}`);
  }
  const md5 = crypto.createHash('md5').update(fs.readFileSync(filePath)).digest('hex');
  for (const [other, otherHash] of shotMd5s) {
    if (otherHash === md5) {
      throw new Error(`Duplicate screenshot: ${path.basename(filePath)} has the same md5 as ${other} (${md5})`);
    }
  }
  shotMd5s.set(path.basename(filePath), md5);
  console.log(`verified ${filePath} ${meta.width}x${meta.height} md5=${md5}`);
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

async function captureBrowserAskModeProof(page) {
  await openConversation(page);
  await setBrowserAskMode(page);
  const askBtn = page.getByRole('button', { name: /^Ask before the agent uses the workbench browser$/i });
  await askBtn.waitFor({ state: 'visible', timeout: 15000 });
  await askBtn.scrollIntoViewIfNeeded();
  await askBtn.hover();
  const tooltip = page.getByRole('tooltip').filter({ hasText: /Ask before the agent uses the workbench browser/i });
  await tooltip.waitFor({ state: 'visible', timeout: 8000 });
  const composer = page.getByPlaceholder(/Ask a question about your documents/i).first();
  await composer.scrollIntoViewIfNeeded();
  await page.waitForTimeout(600);
  await shot(page, 'e2e-02-browser-access-ask-mode-1440x900.png');
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
await captureBrowserAskModeProof(page);

const prompt =
  'Use the shared workbench browser: open https://example.com and answer in one sentence what this page says is for (quote the line from the page body that explains the domain’s purpose—not the document title). I have no uploaded documents about this page.';
await openConversation(page);
await sendChat(page, prompt);

let approve;
try {
  approve = await waitForApproval(page);
} catch (error) {
  await shot(page, 'e2e-failure-before-approval-1440x900.png');
  throw error;
}
const urlPreview = page.getByTestId('tool-approval-argument-preview').filter({ hasText: /https:\/\/example\.com/i });
await urlPreview.waitFor({ state: 'visible', timeout: 30000 });
await page.getByTestId('review-decision-footer').waitFor({ state: 'visible', timeout: 30000 });
await page.getByText(/Approve browser_navigate/i).scrollIntoViewIfNeeded();
await urlPreview.scrollIntoViewIfNeeded();
await page.waitForTimeout(400);
await shot(page, 'e2e-03-tool-approval-card-1440x900.png');
await page.mouse.move(40, 120);
await page.keyboard.press('Escape');
const approveInFooter = page
  .getByTestId('review-decision-footer')
  .getByRole('button', { name: /^Approve once$/i })
  .first();
await page.evaluate(() => {
  document.querySelectorAll('[role="tooltip"]').forEach((node) => node.remove());
});
const [resumeResponse] = await Promise.all([
  page.waitForResponse(
    (response) => response.url().includes('/api/agent-runs/') && response.url().includes('/resume'),
    { timeout: 120000 },
  ),
  approveInFooter.evaluate((button) => {
    (button).dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
  }),
]);
if (!resumeResponse.ok()) {
  throw new Error(`Agent resume failed: HTTP ${resumeResponse.status()}`);
}
await page.waitForTimeout(500);
await page.getByTestId('review-decision-footer').waitFor({ state: 'hidden', timeout: 240000 }).catch(() => {});
await page.getByRole('button', { name: /^Approve once$/i }).first().waitFor({ state: 'hidden', timeout: 240000 }).catch(() => {});
await waitForCapturePage('example\\.com');
await openBrowserTab(page);
const streamFrame = page.locator('iframe[title="Browser"]');
await streamFrame.waitFor({ state: 'visible', timeout: 60000 });
await streamFrame.click({ position: { x: 520, y: 360 }, force: true });
await page.waitForTimeout(45000);
const waitingStream = await page.getByText(/Waiting for stream/i).isVisible().catch(() => false);
if (waitingStream) {
  await page.evaluate(() => {
    const frame = document.querySelector('iframe[title="Browser"]');
    if (frame instanceof HTMLIFrameElement) {
      const src = frame.src;
      frame.src = '';
      frame.src = src;
    }
  });
  await page.waitForTimeout(60000);
}
await shot(page, 'e2e-06-single-agent-browser-tab-1440x900.png');
await openConversation(page);

const bodyQuote = /documentation examples|without needing permission|illustrative examples/i;
const answerDeadline = Date.now() + 600000;
let answerVisible = false;
while (Date.now() < answerDeadline) {
  const candidates = page.locator('div.MuiPaper-root').filter({ hasText: bodyQuote });
  if ((await candidates.count()) > 0) {
    const last = candidates.last();
    if (await last.isVisible().catch(() => false)) {
      answerVisible = true;
      break;
    }
  }
  await page.waitForTimeout(2500);
}
if (!answerVisible) {
  await shot(page, 'e2e-failure-missing-body-answer-1440x900.png');
  throw new Error('Timed out waiting for an answer quoting example.com body text.');
}
await page.waitForTimeout(1500);
await shot(page, 'e2e-04-agent-answer-from-page-1440x900.png');

await sendChat(
  page,
  'Save the current workbench browser page to this thread as a document, then confirm the file name.',
);
await waitForApproval(page, 420000);
const saveUrlPreview = page
  .getByTestId('tool-approval-argument-preview')
  .filter({ hasText: /Save current browser page/i });
await saveUrlPreview.waitFor({ state: 'visible', timeout: 120000 });
await saveUrlPreview.scrollIntoViewIfNeeded();
await shot(page, 'e2e-03b-save-tool-approval-card-1440x900.png');
const saveApprove = page.getByTestId('review-decision-footer').getByRole('button', { name: /^Approve once$/i }).first();
const [saveResumeResponse] = await Promise.all([
  page.waitForResponse(
    (response) => response.url().includes('/api/agent-runs/') && response.url().includes('/resume'),
    { timeout: 240000 },
  ),
  saveApprove.evaluate((button) => {
    (button).dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
  }),
]);
if (!saveResumeResponse.ok()) {
  throw new Error(`Save resume failed: HTTP ${saveResumeResponse.status()}`);
}
await page
  .locator('div.MuiPaper-root')
  .filter({ hasText: /Saved browser page to thread/i })
  .last()
  .waitFor({ state: 'visible', timeout: 420000 });
await page.waitForTimeout(20000);
await openConversation(page);
const docsTab = page.getByRole('tab', { name: /^Documents$/i }).first();
await docsTab.click();
await page.waitForTimeout(3000);
const savedDoc = page
  .locator('[aria-label="Open documents"]')
  .getByRole('tab', { name: /Example Domain/i })
  .first();
await savedDoc.waitFor({ state: 'visible', timeout: 240000 });
await shot(page, 'e2e-05-page-saved-to-thread-1440x900.png');

await context.close();
await browser.close();
console.log('browser-flow-e2e complete');
console.log(JSON.stringify(Object.fromEntries(shotMd5s), null, 2));