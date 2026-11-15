#!/usr/bin/env node
import { chromium } from 'playwright-core';
import { mkdirSync, readFileSync } from 'node:fs';
import path from 'node:path';

const OUT = '/opt/cursor/artifacts';
const APP = process.env.ASKPDF_APP_URL || 'http://127.0.0.1:3000';
const API = process.env.ASKPDF_API_URL || 'http://127.0.0.1:8000/api';
const TOKEN = process.env.ASKPDF_ADMIN_TOKEN;
const PDF = '/agent/repos/askPDF/rag_service/tests/01030000000000.pdf';

async function main() {
  const stamp = Date.now();
  const thread = await (await fetch(`${API}/threads`, { method: 'POST', headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' }, body: JSON.stringify({ name: `clarify-cancel-${stamp}` }) })).json();
  const form = new FormData();
  form.append('file', new Blob([readFileSync(PDF)], { type: 'application/pdf' }), 'sample.pdf');
  await fetch(`${API}/threads/${thread.id}/files/upload`, { method: 'POST', headers: { Authorization: `Bearer ${TOKEN}` }, body: form });
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.addInitScript(({ token }) => { window.localStorage.setItem('auth_token', token); window.localStorage.setItem('last_llm_model', 'inclusionai/ling-3.2-flash:free'); }, { token: TOKEN });
  await page.goto(`${APP}/`, { waitUntil: 'networkidle' });
  await page.getByText(`clarify-cancel-${stamp}`).first().click();
  const input = page.getByPlaceholder('Ask a question about your documents...', { exact: false });
  await input.waitFor({ state: 'visible', timeout: 180000 });
  await input.fill('Compare them.');
  await page.getByRole('button', { name: /send/i }).first().click();
  await page.getByText('I need a bit more clarification').waitFor({ timeout: 420000 });
  await page.getByLabel('Close decision panel').click();
  const value = await input.inputValue();
  if (!value.includes('Compare them.')) throw new Error(`reseed failed: ${value}`);
  await page.screenshot({ path: path.join(OUT, 'clarification-live-cancel.png') });
  await browser.close();
}
main().catch((e) => { console.error(e); process.exit(1); });