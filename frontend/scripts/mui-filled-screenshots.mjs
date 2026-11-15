#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';
import sharp from 'sharp';

const BASE = process.env.ASKPDF_E2E_BASE_URL || 'http://127.0.0.1:3000';
const ART = process.env.ASKPDF_E2E_ARTIFACTS || '/opt/cursor/artifacts';
const PHASE = process.env.SCREENSHOT_PHASE || 'before';
const MODEL = process.env.ASKPDF_E2E_LLM_MODEL || 'google/gemma-4-31b-it:free';

const VIEWPORTS = [
  { name: '1440x900', width: 1440, height: 900 },
  { name: '1024x768', width: 1024, height: 768 },
];

fs.mkdirSync(ART, { recursive: true });

async function verifyShot(filePath, expectedWidth) {
  const meta = await sharp(filePath).metadata();
  if (meta.width !== expectedWidth) {
    throw new Error(`${filePath}: expected width ${expectedWidth}, got ${meta.width}`);
  }
  console.log(`verified ${filePath} ${meta.width}x${meta.height}`);
}

async function shot(page, file, expectedWidth) {
  const filePath = path.join(ART, file);
  await page.screenshot({ path: filePath, fullPage: false });
  await verifyShot(filePath, expectedWidth);
}

async function ensureThread(page) {
  await page.goto(BASE, { waitUntil: 'domcontentloaded', timeout: 120000 });
  await page.waitForTimeout(1200);

  const llm = page.getByRole('combobox', { name: /Select LLM/i }).first();
  if (await llm.isVisible({ timeout: 3000 }).catch(() => false)) {
    return;
  }

  const createProject = page.getByRole('button', { name: /Create Project/i });
  if (await createProject.isVisible({ timeout: 5000 }).catch(() => false)) {
    await createProject.click();
    const projectName = `filled-ui-${Date.now()}`;
    await page.getByRole('textbox').first().fill(projectName);
    await page.getByRole('button', { name: /^Create Project$/i }).click();
    await page.waitForTimeout(1000);
  }

  const createThread = page.getByRole('button', { name: /Create Thread/i }).first();
  await createThread.waitFor({ timeout: 60000 });
  await createThread.click();
  await page.getByRole('textbox').first().fill(`thread-${Date.now()}`);
  await page.getByRole('button', { name: /^Create$/i }).click();
  await llm.waitFor({ timeout: 60000 });
}

async function pickModel(page) {
  const combo = page.getByRole('combobox', { name: /Select LLM/i }).first();
  await combo.click();
  const token = MODEL.split('/').pop() || MODEL;
  await combo.fill(token.replace(/:free$/, ''));
  await page.waitForTimeout(500);
  const option = page.getByRole('option', { name: new RegExp(MODEL.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i') }).first();
  if (await option.count()) {
    await option.click();
  } else {
    await page.keyboard.press('Escape');
  }
}

async function openChatTab(page) {
  const chatTab = page.getByRole('tab', { name: /^Chat$/i }).first();
  if (await chatTab.isVisible({ timeout: 3000 }).catch(() => false)) {
    await chatTab.click();
    await page.waitForTimeout(400);
  }
}

async function captureThreadHeaderComposer(page, vp) {
  await ensureThread(page);
  await pickModel(page);
  await openChatTab(page);
  const composer = page.getByPlaceholder(/Ask a question about your documents/i).first();
  await composer.waitFor({ timeout: 60000 });
  const header = page.getByRole('combobox', { name: /Select LLM/i }).first();
  await header.waitFor({ timeout: 10000 });
  const box = await header.evaluate((el) => {
    const r = el.getBoundingClientRect();
    return {
      x: 0,
      y: Math.max(0, r.top - 12),
      width: window.innerWidth,
      height: Math.min(window.innerHeight, 520),
    };
  });
  await page.screenshot({
    path: path.join(ART, `${PHASE}-thread-header-composer-${vp.name}.png`),
    clip: box,
  });
  await verifyShot(path.join(ART, `${PHASE}-thread-header-composer-${vp.name}.png`), vp.width);
}

async function captureSettingsDialog(page, vp) {
  await ensureThread(page);
  await openChatTab(page);
  const settings = page.getByRole('button', { name: /AI prompt settings for this thread/i }).first();
  await settings.waitFor({ timeout: 20000 });
  await settings.click();
  const dialog = page.getByRole('dialog');
  await dialog.waitFor({ timeout: 15000 });
  await dialog.getByRole('combobox', { name: /^Agent workflow/i }).waitFor({ timeout: 15000 });
  await page.waitForTimeout(400);
  await shot(page, `${PHASE}-settings-agent-workflow-${vp.name}.png`, vp.width);
  await page.keyboard.press('Escape');
}

async function captureWorkflowToolbar(page, vp) {
  await page.goto(`${BASE}/agent-workflow-builder`, { waitUntil: 'domcontentloaded', timeout: 120000 });
  await page.waitForTimeout(2000);
  await page.getByRole('combobox', { name: /Workflow/i }).first().waitFor({ timeout: 60000 });
  await shot(page, `${PHASE}-workflow-toolbar-${vp.name}.png`, vp.width);
}

async function captureEmbeddingsSelects(page, vp) {
  await ensureThread(page);
  await page.getByRole('tab', { name: 'Documents', exact: true }).first().click();
  await page.waitForTimeout(400);
  await page.getByRole('tab', { name: 'Embeddings', exact: true }).first().click();
  await page.getByLabel('Filter chunks').first().waitFor({ timeout: 120000 });
  const refresh = page.getByRole('button', { name: 'Refresh' }).first();
  await refresh.waitFor({ timeout: 30000 });
  await page.waitForTimeout(1200);
  const chrome = await refresh.evaluate((el) => {
    const r = el.getBoundingClientRect();
    return {
      x: 0,
      y: Math.max(0, r.top - 10),
      width: window.innerWidth,
      height: Math.min(window.innerHeight - Math.max(0, r.top - 10), 200),
    };
  });
  await page.screenshot({
    path: path.join(ART, `${PHASE}-embeddings-selects-${vp.name}.png`),
    clip: chrome,
  });
  await verifyShot(path.join(ART, `${PHASE}-embeddings-selects-${vp.name}.png`), vp.width);
}

const browser = await chromium.launch({ headless: true });
for (const vp of VIEWPORTS) {
  const context = await browser.newContext({ viewport: { width: vp.width, height: vp.height } });
  const page = await context.newPage();
  await captureThreadHeaderComposer(page, vp);
  await captureSettingsDialog(page, vp);
  await captureWorkflowToolbar(page, vp);
  await captureEmbeddingsSelects(page, vp);
  await context.close();
}
await browser.close();