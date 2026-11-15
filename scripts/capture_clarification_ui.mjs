#!/usr/bin/env node
/**
 * Capture 1440x900 screenshots of the clarification HITL card on the live app.
 * Uses a one-shot chat response override when the model does not emit clarify naturally.
 */
import { chromium } from 'playwright-core';
import { mkdirSync } from 'node:fs';
import path from 'node:path';

const VIEWPORT = { width: 1440, height: 900 };
const OUT = '/opt/cursor/artifacts';
const APP = process.env.ASKPDF_APP_URL || 'http://127.0.0.1:3000';
const API = process.env.ASKPDF_API_URL || 'http://127.0.0.1:8000/api';
const TOKEN = process.env.ASKPDF_ADMIN_TOKEN;

mkdirSync(OUT, { recursive: true });

async function api(pathname, init = {}) {
  const headers = {
    Authorization: `Bearer ${TOKEN}`,
    'Content-Type': 'application/json',
    ...(init.headers || {}),
  };
  const res = await fetch(`${API}${pathname}`, { ...init, headers });
  if (!res.ok) throw new Error(`${pathname} ${res.status} ${await res.text()}`);
  return res.json();
}

const clarificationPayload = {
  status: 'clarification_required',
  clarification_options: [
    'What is the content of the uploaded PDF file?',
    'What are all the sources available in this thread, including web search results?',
  ],
  answer: '',
  agent_run_id: null,
};

async function main() {
  if (!TOKEN) throw new Error('ASKPDF_ADMIN_TOKEN required');
  const thread = await api('/threads', { method: 'POST', body: JSON.stringify({ name: 'clarification-ui-proof' }) });
  const threadId = thread.id;

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: VIEWPORT });
  let interceptOnce = true;

  await page.route(`**/${threadId}/chat`, async (route) => {
    if (route.request().method() !== 'POST' || !interceptOnce) {
      await route.continue();
      return;
    }
    interceptOnce = false;
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(clarificationPayload),
    });
  });

  await page.goto(`${APP}/?thread=${threadId}`, { waitUntil: 'networkidle' });
  const input = page.locator('textarea').first();
  await input.click();
  await input.fill('what documents are available?');
  await page.getByRole('button', { name: /send/i }).first().click();
  await page.getByText('I need a bit more clarification').waitFor({ timeout: 15000 });
  await page.screenshot({ path: path.join(OUT, 'clarification-card-appears.png'), fullPage: false });

  await page.getByRole('button', { name: /What is the content of the uploaded PDF file/i }).click();
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, 'clarification-option-clicked.png'), fullPage: false });

  interceptOnce = true;
  await page.route(`**/${threadId}/chat`, async (route) => {
    if (route.request().method() !== 'POST' || !interceptOnce) {
      await route.continue();
      return;
    }
    interceptOnce = false;
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'completed',
        answer: 'The uploaded PDF discusses multi-head attention in Transformer models.',
        user_message_id: 'proof-user-2',
        assistant_message_id: 'proof-asst-2',
      }),
    });
  });
  await input.fill('what documents are available?');
  await page.getByRole('button', { name: /send/i }).first().click();
  await page.getByText('multi-head attention').waitFor({ timeout: 15000 });
  await page.screenshot({ path: path.join(OUT, 'clarification-after-option-answer.png'), fullPage: false });

  interceptOnce = true;
  await page.route(`**/${threadId}/chat`, async (route) => {
    if (route.request().method() !== 'POST' || !interceptOnce) {
      await route.continue();
      return;
    }
    interceptOnce = false;
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(clarificationPayload),
    });
  });
  await input.fill('what documents are available?');
  await page.getByRole('button', { name: /send/i }).first().click();
  await page.getByText('I need a bit more clarification').waitFor({ timeout: 15000 });
  await page.getByRole('button', { name: /Something else/i }).click();
  const custom = page.getByLabel('Your response');
  await custom.fill('List each filename attached to this thread.');
  await page.getByLabel('Send your response').click();
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, 'clarification-custom-question.png'), fullPage: false });

  interceptOnce = true;
  await page.route(`**/${threadId}/chat`, async (route) => {
    if (route.request().method() !== 'POST' || !interceptOnce) {
      await route.continue();
      return;
    }
    interceptOnce = false;
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'completed',
        answer: 'This thread has one attached file: attention.pdf.',
        user_message_id: 'proof-user-3',
        assistant_message_id: 'proof-asst-3',
      }),
    });
  });
  await page.waitForTimeout(1200);
  await page.screenshot({ path: path.join(OUT, 'clarification-custom-answer.png'), fullPage: false });

  interceptOnce = true;
  await page.route(`**/${threadId}/chat`, async (route) => {
    if (route.request().method() !== 'POST' || !interceptOnce) {
      await route.continue();
      return;
    }
    interceptOnce = false;
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(clarificationPayload),
    });
  });
  await input.fill('what documents are available?');
  await page.getByRole('button', { name: /send/i }).first().click();
  await page.getByText('I need a bit more clarification').waitFor({ timeout: 15000 });
  await page.getByLabel('Close decision panel').click();
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, 'clarification-after-cancel.png'), fullPage: false });

  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});