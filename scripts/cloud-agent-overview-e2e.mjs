#!/usr/bin/env node
/** API + Playwright screenshot: overview question with gpt-4.1-mini on Attention PDF. */
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import {
  DEFAULT_API_URL,
  DEFAULT_ART_DIR,
  DEFAULT_BASE_URL,
  REPO_ROOT,
  apiJson,
  defaultProjectId,
  loadEnvAdmin,
  waitForFileReady,
} from './cloud-agent-e2e-lib.mjs';

const requireFromFrontend = createRequire(path.join(REPO_ROOT, 'frontend/package.json'));
const { chromium } = requireFromFrontend('@playwright/test');

const MODEL = process.env.ASKPDF_E2E_LLM_MODEL || 'openai/gpt-4.1-mini';
const PDF_PATH =
  process.env.ASKPDF_E2E_PDF || path.join(REPO_ROOT, '.benchmark-pdf-cache/1706.03762.pdf');

async function main() {
  fs.mkdirSync(DEFAULT_ART_DIR, { recursive: true });
  const admin = loadEnvAdmin();
  const projectId = defaultProjectId(await apiJson(`${DEFAULT_API_URL}/api/projects`, { admin }));
  const threadName = `overview-ui-proof-${Date.now()}`;
  const thread = await apiJson(`${DEFAULT_API_URL}/api/threads`, {
    method: 'POST',
    admin,
    body: { name: threadName, project_id: projectId },
  });
  await apiJson(`${DEFAULT_API_URL}/api/threads/${thread.id}/settings`, {
    method: 'PUT',
    admin,
    body: { agent_workflow: { workflow_id: 'auto_rag_agent' } },
  });
  const form = new FormData();
  form.append('file', new Blob([fs.readFileSync(PDF_PATH)]), path.basename(PDF_PATH));
  const uploadRes = await fetch(`${DEFAULT_API_URL}/api/threads/${thread.id}/files/upload`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${admin}` },
    body: form,
  });
  if (!uploadRes.ok) throw new Error(`upload ${uploadRes.status}`);
  const uploaded = await uploadRes.json();
  await waitForFileReady(thread.id, uploaded.file_hash, admin);
  const chat = await apiJson(`${DEFAULT_API_URL}/api/threads/${thread.id}/chat`, {
    method: 'POST',
    admin,
    body: {
      thread_id: thread.id,
      question: 'What does this paper discuss?',
      llm_model: MODEL,
      bypass_clarification: true,
    },
  });
  console.log('[overview-e2e] answer preview:', String(chat.answer || '').slice(0, 280));
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
  await page.goto(DEFAULT_BASE_URL, { waitUntil: 'networkidle', timeout: 120000 });
  await page.getByText(threadName, { exact: false }).first().click({ timeout: 120000 });
  await page.getByText('What does this paper discuss?', { exact: false }).first().waitFor({ timeout: 120000 });
  const shot = path.join(DEFAULT_ART_DIR, 'overview-answer-gpt41mini.png');
  await page.screenshot({ path: shot, fullPage: false });
  console.log('[overview-e2e] screenshot', shot);
  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});