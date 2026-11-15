#!/usr/bin/env node
/**
 * API setup + Playwright screenshot proving KaTeX-rendered chat math (requires stack).
 */
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import {
  DEFAULT_API_URL,
  DEFAULT_ART_DIR,
  DEFAULT_BASE_URL,
  REPO_ROOT,
  apiJson,
  loadEnvAdmin,
  waitForFileReady,
} from './cloud-agent-e2e-lib.mjs';

const requireFromFrontend = createRequire(path.join(REPO_ROOT, 'frontend/package.json'));
const { chromium } = requireFromFrontend('@playwright/test');

const MODEL = process.env.ASKPDF_E2E_LLM_MODEL || 'openai/gpt-4.1-mini';
const PDF_PATH =
  process.env.ASKPDF_E2E_ATTENTION_PDF
  || path.join(REPO_ROOT, '.cache-e2e', 'Attention.pdf');

function ensureAttentionPdf() {
  if (fs.existsSync(PDF_PATH)) return;
  fs.mkdirSync(path.dirname(PDF_PATH), { recursive: true });
  const r = spawnSync(
    'curl',
    ['-fsSL', 'https://arxiv.org/pdf/1706.03762.pdf', '-o', PDF_PATH],
    { stdio: 'inherit' },
  );
  if (r.status !== 0) throw new Error('failed to download Attention.pdf');
}

async function main() {
  fs.mkdirSync(DEFAULT_ART_DIR, { recursive: true });
  ensureAttentionPdf();
  const admin = loadEnvAdmin();

  const project = await apiJson(`${DEFAULT_API_URL}/api/projects`, {
    method: 'POST',
    admin,
    body: {
      name: `math-proof-${Date.now()}`,
      description: 'KaTeX chat proof',
      embedding_model: 'BAAI/bge-m3',
    },
  });

  const thread = await apiJson(`${DEFAULT_API_URL}/api/projects/${project.id}/threads`, {
    method: 'POST',
    admin,
    body: { name: 'attention-math-proof' },
  });

  const form = new FormData();
  form.append('file', new Blob([fs.readFileSync(PDF_PATH)]), 'Attention.pdf');
  const uploadRes = await fetch(`${DEFAULT_API_URL}/api/threads/${thread.id}/files/upload`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${admin}` },
    body: form,
  });
  if (!uploadRes.ok) throw new Error(`upload failed: ${uploadRes.status} ${await uploadRes.text()}`);
  const uploaded = await uploadRes.json();

  await waitForFileReady(thread.id, uploaded.file_hash, admin);

  const chat = await apiJson(`${DEFAULT_API_URL}/api/threads/${thread.id}/chat`, {
    method: 'POST',
    admin,
    body: {
      thread_id: thread.id,
      question: 'What is the scaled dot-product attention mechanism in the Attention paper?',
      llm_model: MODEL,
    },
  });

  const answer = String(chat?.answer || chat?.content || '');
  console.log('[math-e2e] assistant preview:', answer.slice(0, 280));
  if (answer.length < 40) throw new Error('chat answer too short');

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.goto(DEFAULT_BASE_URL, { waitUntil: 'domcontentloaded', timeout: 120000 });
  await page.waitForTimeout(2000);

  const projectRow = page.getByText(project.name, { exact: false }).first();
  if (await projectRow.count()) {
    await projectRow.click();
    await page.waitForTimeout(1500);
  }

  const threadRow = page.getByText('attention-math-proof', { exact: false }).first();
  await threadRow.click({ timeout: 60000 });
  await page.waitForTimeout(2500);

  await page.locator('.katex').first().waitFor({ timeout: 120000 });

  const shot = path.join(DEFAULT_ART_DIR, 'chat-math-rendered.png');
  await page.screenshot({ path: shot, fullPage: false });
  console.log('[math-e2e] screenshot', shot);
  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});