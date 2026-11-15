#!/usr/bin/env node
/**
 * Real clarification HITL UI proof (no staged SSE). Requires running stack + OpenRouter key.
 */
import { chromium } from 'playwright-core';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, '../..');
const VIEWPORT = { width: 1440, height: 900 };
const OUT = '/opt/cursor/artifacts';
const APP = process.env.ASKPDF_APP_URL || 'http://127.0.0.1:3000';
const API = process.env.ASKPDF_API_URL || 'http://127.0.0.1:8000/api';
const TOKEN = process.env.ASKPDF_ADMIN_TOKEN;
const MODEL = process.env.ASKPDF_CLARIFY_MODEL || 'inclusionai/ling-3.2-flash:free';
const PDF = process.env.ASKPDF_PROOF_PDF || path.join(REPO, 'rag_service/tests/01030000000000.pdf');
const AMBIGUOUS = process.env.ASKPDF_CLARIFY_QUESTION || 'Compare them.';

mkdirSync(OUT, { recursive: true });

async function api(pathname, init = {}) {
  const headers = {
    Authorization: `Bearer ${TOKEN}`,
    ...(init.body && !(init.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}),
    ...(init.headers || {}),
  };
  const res = await fetch(`${API}${pathname}`, { ...init, headers });
  if (!res.ok) throw new Error(`${pathname} ${res.status} ${await res.text()}`);
  if (res.status === 204) return null;
  return res.json();
}

async function waitIndex(threadId) {
  for (let i = 0; i < 120; i += 1) {
    const st = await api(`/threads/${threadId}/indexing/status`);
    if (st.status === 'ready' && st.embedding_model_ready !== false) return st;
    await new Promise((r) => setTimeout(r, 2000));
  }
  throw new Error('indexing not ready');
}

async function waitComposerReady(page) {
  for (let attempt = 0; attempt < 600; attempt += 1) {
    const sending = await page.getByPlaceholder('Sending...', { exact: false }).count();
    const chatInput = page.getByPlaceholder('Ask a question about your documents...', { exact: false });
    const visible = await chatInput.isVisible().catch(() => false);
    if (sending === 0 && visible && await chatInput.isEnabled()) {
      return chatInput;
    }
    await page.waitForTimeout(1000);
  }
  throw new Error('Composer did not become ready');
}

async function main() {
  if (!TOKEN) throw new Error('ASKPDF_ADMIN_TOKEN required');
  const stamp = Date.now();
  const thread = await api('/threads', { method: 'POST', body: JSON.stringify({ name: `clarify-live-${stamp}` }) });
  const threadId = thread.id;
  const pdfBuf = readFileSync(PDF);
  const form = new FormData();
  form.append('file', new Blob([pdfBuf], { type: 'application/pdf' }), 'sample.pdf');
  const up = await fetch(`${API}/threads/${threadId}/files/upload`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${TOKEN}` },
    body: form,
  });
  if (!up.ok) throw new Error(`upload ${up.status} ${await up.text()}`);
  await waitIndex(threadId);

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: VIEWPORT });
  await page.addInitScript(({ token, model }) => {
    window.localStorage.setItem('auth_token', token);
    window.localStorage.setItem('last_llm_model', model);
  }, { token: TOKEN, model: MODEL });

  await page.goto(`${APP}/`, { waitUntil: 'networkidle', timeout: 120000 });
  await page.getByText(`clarify-live-${stamp}`, { exact: false }).first().click({ timeout: 60000 });

  let chatInput = await waitComposerReady(page);
  await chatInput.fill(AMBIGUOUS);
  await page.getByRole('button', { name: /send/i }).first().click();

  await page.getByText('I need a bit more clarification').waitFor({ timeout: 300000 });
  await page.getByRole('button', { name: /Something else/i }).waitFor({ timeout: 30000 });
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, 'clarification-live-card.png') });

  await page.getByRole('button', { name: /Something else/i }).click();
  const customText = 'Summarize the architecture section in one paragraph.';
  await page.getByRole('textbox', { name: 'Your response' }).fill(customText);
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, 'clarification-live-custom-box.png') });
  await page.getByRole('button', { name: 'Cancel custom response' }).click();
  await page.waitForTimeout(300);

  const firstOption = page.getByRole('button').filter({ hasText: /Compare|MORAN/i }).first();
  const optionLabel = (await firstOption.textContent())?.trim() || '';
  await firstOption.click();
  await page.getByPlaceholder('Sending...', { exact: false }).waitFor({ timeout: 120000 });
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, 'clarification-live-option-clicked.png') });
  await waitComposerReady(page);

  chatInput = await waitComposerReady(page);
  await chatInput.fill(AMBIGUOUS);
  await page.getByRole('button', { name: /send/i }).first().click();
  await page.getByText('I need a bit more clarification').waitFor({ timeout: 300000 });
  await page.getByLabel('Close decision panel').click();
  await page.waitForTimeout(800);
  chatInput = await waitComposerReady(page);
  const composerValue = await chatInput.inputValue();
  if (!composerValue.includes(AMBIGUOUS)) {
    throw new Error(`Cancel did not re-seed composer: "${composerValue}"`);
  }
  await page.screenshot({ path: path.join(OUT, 'clarification-live-cancel.png') });

  writeFileSync(
    path.join(OUT, 'clarification-live-ui.json'),
    JSON.stringify({ thread_id: threadId, model: MODEL, question: AMBIGUOUS, picked_option: optionLabel, custom_question: customText }, null, 2),
  );

  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});