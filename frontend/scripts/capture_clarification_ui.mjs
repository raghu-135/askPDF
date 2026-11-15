#!/usr/bin/env node
import { chromium } from 'playwright-core';
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
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

async function expectEnabled(locator) {
  await locator.waitFor({ state: 'attached', timeout: 60000 });
  for (let attempt = 0; attempt < 120; attempt += 1) {
    if (await locator.isEnabled()) return;
    await locator.page().waitForTimeout(500);
  }
  throw new Error('Composer did not become enabled');
}

async function main() {
  if (!TOKEN) throw new Error('ASKPDF_ADMIN_TOKEN required');
  const thread = await api('/threads', { method: 'POST', body: JSON.stringify({ name: 'clarification-ui-proof' }) });
  const threadId = thread.id;

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: VIEWPORT });
  await page.route('**/api/models', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        llm_models: ['inclusionai/ling-3.2-flash:free'],
        all_models: [{ id: 'inclusionai/ling-3.2-flash:free', supports_tools: true }],
      }),
    });
  });
  await page.route('**/api/threads/*/indexing/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ thread_id: threadId, status: 'ready', embedding_model_ready: true, stats: {} }),
    });
  });
  await page.route('**/api/health/embed-model/**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ model: 'BAAI/bge-m3', embedding_model_ready: true }),
    });
  });
  await page.route('**/api/health/chat-model/**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        model: 'inclusionai/ling-3.2-flash:free',
        chat_model_ready: true,
        supports_tools: true,
        can_invoke_tools: true,
      }),
    });
  });
  let interceptOnce = true;

  const fulfillStream = async (route, responsePayload) => {
    const envelope = {
      id: 1,
      event: responsePayload.clarification_options ? 'run.clarification' : 'run.completed',
      data: { response: responsePayload },
    };
    const body = `data: ${JSON.stringify(envelope)}\n\n`;
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'text/event-stream; charset=utf-8' },
      body,
    });
  };

  await page.route('**/threads/*/chat', async (route) => {
    if (!route.request().url().includes(threadId) || route.request().method() !== 'POST') {
      await route.continue();
      return;
    }
    if (!interceptOnce) {
      await route.continue();
      return;
    }
    interceptOnce = false;
    await fulfillStream(route, clarificationPayload);
  });

  await page.addInitScript(() => {
    window.localStorage.setItem('last_llm_model', 'inclusionai/ling-3.2-flash:free');
  });
  await page.goto(`${APP}/`, { waitUntil: 'networkidle' });
  await page.getByText('clarification-ui-proof', { exact: false }).first().click({ timeout: 15000 });
  const chatInput = page.getByPlaceholder('Ask a question about your documents...', { exact: false });
  await chatInput.waitFor({ state: 'visible', timeout: 120000 });
  await expectEnabled(chatInput);
  await chatInput.click({ timeout: 15000 });
  await chatInput.fill('what documents are available?');
  await page.getByRole('button', { name: /send/i }).first().click();
  await page.getByText('I need a bit more clarification').waitFor({ timeout: 20000 });
  await page.screenshot({ path: path.join(OUT, 'clarification-card-appears.png') });

  interceptOnce = true;
  await page.unroute('**/threads/*/chat');
  await page.route('**/threads/*/chat', async (route) => {
    if (!route.request().url().includes(threadId) || route.request().method() !== 'POST') {
      await route.continue();
      return;
    }
    if (!interceptOnce) {
      await route.continue();
      return;
    }
    interceptOnce = false;
  await fulfillStream(route, {
      status: 'completed',
      answer: 'The uploaded PDF discusses multi-head attention in Transformer models.',
    });
  });
  await page.getByRole('button', { name: /What is the content of the uploaded PDF file/i }).click();
  await page.getByText('multi-head attention', { timeout: 20000 }).waitFor();
  await page.screenshot({ path: path.join(OUT, 'clarification-option-clicked.png') });

  interceptOnce = true;
  await page.unroute('**/threads/*/chat');
  await page.route('**/threads/*/chat', async (route) => {
    if (!route.request().url().includes(threadId) || route.request().method() !== 'POST') {
      await route.continue();
      return;
    }
    if (!interceptOnce) {
      await route.continue();
      return;
    }
    interceptOnce = false;
    await fulfillStream(route, clarificationPayload);
  });
  await chatInput.fill('what documents are available?');
  await page.getByRole('button', { name: /send/i }).first().click();
  await page.getByText('I need a bit more clarification').waitFor({ timeout: 20000 });
  await page.getByRole('button', { name: /Something else/i }).click();
  await page.getByRole('textbox', { name: 'Your response' }).fill('List each filename attached to this thread.');
  interceptOnce = true;
  await page.unroute('**/threads/*/chat');
  await page.route('**/threads/*/chat', async (route) => {
    if (!route.request().url().includes(threadId) || route.request().method() !== 'POST') {
      await route.continue();
      return;
    }
    if (!interceptOnce) {
      await route.continue();
      return;
    }
    interceptOnce = false;
    await fulfillStream(route, {
      status: 'completed',
      answer: 'This thread has one attached file: attention.pdf.',
    });
  });
  await page.getByLabel('Send your response').click();
  await page.getByText('attention.pdf', { timeout: 20000 }).waitFor();
  await page.screenshot({ path: path.join(OUT, 'clarification-custom-question.png') });

  interceptOnce = true;
  await page.unroute('**/threads/*/chat');
  await page.route('**/threads/*/chat', async (route) => {
    if (!route.request().url().includes(threadId) || route.request().method() !== 'POST') {
      await route.continue();
      return;
    }
    if (!interceptOnce) {
      await route.continue();
      return;
    }
    interceptOnce = false;
    await fulfillStream(route, clarificationPayload);
  });
  await chatInput.fill('what documents are available?');
  await page.getByRole('button', { name: /send/i }).first().click();
  await page.getByText('I need a bit more clarification').waitFor({ timeout: 20000 });
  await page.getByLabel('Close decision panel').click();
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, 'clarification-after-cancel.png') });

  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});