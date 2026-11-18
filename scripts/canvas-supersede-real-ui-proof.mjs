#!/usr/bin/env node
/**
 * Single-thread UI proof: create canvas → quote link → simple edit → same quoted id, new revision.
 */
import { spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import {
  DEFAULT_API_URL,
  DEFAULT_ART_DIR,
  DEFAULT_BASE_URL,
  REPO_ROOT,
  REAL_UI_VIEWPORT,
  ScreenshotGuard,
  apiJson,
  chatRateLimitPause,
  ensureProjectAndThread,
  loadEnvAdmin,
  selectLlmModel,
  typeAndSendChat,
  waitForAssistantAnswer,
  waitForComposerReady,
  waitForNoLoadingShell,
  waitForThreadWorkspaceReady,
} from './cloud-agent-e2e-lib.mjs';

const requireFromFrontend = createRequire(path.join(REPO_ROOT, 'frontend/package.json'));
const { chromium } = requireFromFrontend('@playwright/test');

const MODEL = process.env.ASKPDF_E2E_LLM_MODEL || 'inclusionai/ling-3.2-flash:free';
const ART_DIR = DEFAULT_ART_DIR;
const EDITED_TITLE = 'OpenRouter Free Models (edited)';

function gitSha() {
  const r = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: REPO_ROOT, encoding: 'utf8' });
  return r.status === 0 ? r.stdout.trim() : 'unknown';
}

async function waitForEmbedReady(admin) {
  for (let i = 0; i < 80; i += 1) {
    const res = await fetch(
      `${DEFAULT_API_URL}/api/health/embed-model/${encodeURIComponent('BAAI/bge-m3')}`,
      { headers: { Authorization: `Bearer ${admin}` } },
    );
    const data = await res.json();
    if (data.embedding_model_ready) return;
    await new Promise((r) => setTimeout(r, 3000));
  }
  throw new Error('Embedding model not ready');
}

async function configureThreadForProof(admin, threadName) {
  const payload = await apiJson(`${DEFAULT_API_URL}/api/threads`, { admin });
  const rows = Array.isArray(payload) ? payload : payload.threads || [];
  const thread = rows.find((row) => row.name === threadName);
  if (!thread?.id) throw new Error(`Could not resolve thread id for ${threadName}`);
  await apiJson(`${DEFAULT_API_URL}/api/threads/${thread.id}/settings`, {
    method: 'PUT',
    admin,
    body: {
      agent_workflow: { workflow_id: 'router_rag_agent' },
      hitl_canvas_publish: false,
      hitl_web_approval: false,
      use_web_search: true,
    },
  });
  return thread.id;
}

async function openCanvasInWorkspace(page, admin, threadId, { titlePattern } = {}) {
  const deadline = Date.now() + 600_000;
  while (Date.now() < deadline) {
    const list = await apiJson(`${DEFAULT_API_URL}/api/threads/${threadId}/canvases?current_only=true`, { admin });
    const canvases = list.canvases || [];
    if (canvases.length > 0) {
      const pick = titlePattern
        ? canvases.find((row) => titlePattern.test(String(row.title || ''))) || canvases[0]
        : canvases[0];
      const { id: canvasId, title } = pick;
      const url = `${DEFAULT_BASE_URL}/?thread=${threadId}&canvas=${canvasId}`;
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120_000 });
      await waitForNoLoadingShell(page);
      await waitForThreadWorkspaceReady(page);
      await page.getByRole('heading', { name: title }).first().waitFor({ timeout: 120_000 });
      return { title, canvasId };
    }
    await page.waitForTimeout(5000);
  }
  throw new Error('No canvas was published in this thread');
}

async function resolveCanvasRecord(admin, threadId, canvasId) {
  return apiJson(
    `${DEFAULT_API_URL}/api/threads/${threadId}/canvases/${canvasId}?resolve_current=true`,
    { admin },
  );
}

async function waitForCanvasRevision(admin, threadId, quotedCanvasId, beforeRecord) {
  const quoted = String(quotedCanvasId).toLowerCase();
  const deadline = Date.now() + 480_000;
  while (Date.now() < deadline) {
    const afterRecord = await resolveCanvasRecord(admin, threadId, quotedCanvasId);
    const title = String(afterRecord.title || '');
    const summary = String(afterRecord.summary || afterRecord.spec?.summary || '');
    const changed = afterRecord.id !== beforeRecord.id
      || title !== beforeRecord.title
      || summary !== String(beforeRecord.summary || beforeRecord.spec?.summary || '');
    const list = await apiJson(`${DEFAULT_API_URL}/api/threads/${threadId}/canvases?current_only=false`, { admin });
    const hasChild = (list.canvases || []).some(
      (row) => String(row.supersedes_id || '').toLowerCase() === quoted,
    );
    if ((hasChild || changed) && title.includes('(edited)')) {
      return afterRecord;
    }
    await new Promise((r) => setTimeout(r, 8000));
  }
  throw new Error('Canvas revision was not published after the edit request');
}

function parseCanvasDeepLink(url) {
  const u = new URL(url);
  const threadId = u.searchParams.get('thread');
  const canvasId = u.searchParams.get('canvas');
  if (!threadId || !canvasId) throw new Error(`Clipboard is not a canvas deep link: ${url}`);
  return { threadId, canvasId, url: u.toString() };
}

function evidenceRowCount(spec) {
  const safe = spec && typeof spec === 'object' ? spec : {};
  for (const section of safe.sections || []) {
    for (const block of section.blocks || []) {
      if (block.type === 'table' && Array.isArray(block.rows)) {
        return block.rows.length;
      }
    }
  }
  return 0;
}

async function reloadThreadChatWorkspace(page, deepLinkUrl, lineageHeadCanvasId) {
  const reloadTarget = new URL(deepLinkUrl);
  reloadTarget.searchParams.set('canvas', lineageHeadCanvasId);
  const threadId = reloadTarget.searchParams.get('thread');
  if (!threadId) throw new Error('Deep link missing thread id');

  await page.goto(reloadTarget.toString(), { waitUntil: 'domcontentloaded', timeout: 120_000 });
  await waitForNoLoadingShell(page);
  await waitForThreadWorkspaceReady(page);
  await page.getByTestId('chat-message-list').waitFor({ state: 'visible', timeout: 180_000 });
  await page.getByText('Research canvas', { exact: true }).nth(1).waitFor({ state: 'visible', timeout: 180_000 });
}

async function copyCanvasLinkFromTab(page) {
  const copy = page.locator('[aria-label="Copy canvas link"]').first();
  await copy.waitFor({ state: 'visible', timeout: 120_000 });
  await copy.scrollIntoViewIfNeeded();
  await copy.click();
  await page.locator('[aria-label="Link copied"]').first()
    .waitFor({ state: 'visible', timeout: 15_000 });
  return page.evaluate(() => navigator.clipboard.readText());
}

async function focusChatComposer(page) {
  const composer = page.getByTestId('chat-composer-input');
  if (!(await composer.isVisible({ timeout: 5000 }).catch(() => false))) {
    const overview = page.getByRole('tab', { name: /Overview/i }).first();
    if (await overview.isVisible({ timeout: 5000 }).catch(() => false)) {
      await overview.click();
    }
  }
  await composer.scrollIntoViewIfNeeded();
  await waitForComposerReady(page);
}

async function main() {
  const testedSha = gitSha();
  const admin = loadEnvAdmin();
  await waitForEmbedReady(admin);

  fs.mkdirSync(ART_DIR, { recursive: true });
  const guard = new ScreenshotGuard(ART_DIR);
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: REAL_UI_VIEWPORT,
    permissions: ['clipboard-read', 'clipboard-write'],
  });
  await context.addInitScript(({ model }) => {
    window.localStorage.setItem('last_llm_model', model);
    window.localStorage.setItem('last_use_web_search', '1');
  }, { model: MODEL });
  const page = await context.newPage();

  const shots = {
    before: 'canvas-before-1440x900.png',
    after: 'canvas-after-same-id-1440x900.png',
    reload: 'chat-after-reload-1440x900.png',
  };

  try {
    await page.goto(DEFAULT_BASE_URL, { waitUntil: 'domcontentloaded', timeout: 120_000 });
    const { threadName } = await ensureProjectAndThread(page);
    const uiThreadId = await configureThreadForProof(admin, threadName);
    await selectLlmModel(page, MODEL);
    await waitForThreadWorkspaceReady(page);

    const createPrompt =
      'Please create a brief research canvas about how OpenRouter lists free models. '
      + 'Include at least two distinct claims with web sources.';
    await chatRateLimitPause();
    await typeAndSendChat(page, createPrompt);
    await waitForAssistantAnswer(page);

    const { canvasId: expectedCanvasId } = await openCanvasInWorkspace(page, admin, uiThreadId);
    await guard.capture(page, shots.before, { requireChatWorkspace: true });

    const deepLink = await copyCanvasLinkFromTab(page);
    const { threadId, canvasId } = parseCanvasDeepLink(deepLink);
    if (canvasId !== expectedCanvasId) {
      throw new Error(`Copied canvas id ${canvasId} does not match ${expectedCanvasId}`);
    }

    const beforeRecord = await resolveCanvasRecord(admin, threadId, canvasId);
    const claimsBefore = evidenceRowCount(beforeRecord.spec);
    if (claimsBefore < 2) {
      throw new Error(`Need at least 2 evidence rows before edit (got ${claimsBefore})`);
    }

    const updatePrompt =
      `Using this canvas, rename the title to "${EDITED_TITLE}" and shorten the overview text. `
      + 'Keep the evidence matrix claims unchanged.\n\n'
      + deepLink;
    await focusChatComposer(page);
    await chatRateLimitPause();
    await typeAndSendChat(page, updatePrompt);
    await waitForAssistantAnswer(page);

    let afterRecord;
    try {
      afterRecord = await waitForCanvasRevision(admin, threadId, canvasId, beforeRecord);
    } catch (err) {
      await focusChatComposer(page);
      await typeAndSendChat(
        page,
        `Republish that quoted canvas: title "${EDITED_TITLE}", shorter overview, same evidence rows.\n\n${deepLink}`,
      );
      await chatRateLimitPause();
      await waitForAssistantAnswer(page);
      afterRecord = await waitForCanvasRevision(admin, threadId, canvasId, beforeRecord);
      if (!afterRecord.title?.includes('(edited)')) throw err;
    }

    const afterDeepLink = new URL(deepLink);
    if (afterDeepLink.searchParams.get('canvas') !== canvasId) {
      throw new Error('Quoted canvas id in link changed after edit');
    }

    const claimsAfter = evidenceRowCount(afterRecord.spec);
    if (claimsAfter < claimsBefore) {
      throw new Error(`Evidence rows dropped (${claimsBefore} -> ${claimsAfter})`);
    }

    await openCanvasInWorkspace(page, admin, threadId, { titlePattern: /\(edited\)/i });
    await page.getByRole('heading', { name: afterRecord.title }).first().waitFor({ timeout: 120_000 });
    await guard.capture(page, shots.after, { requireChatWorkspace: true });

    const persisted = await apiJson(`${DEFAULT_API_URL}/api/threads/${threadId}/messages?limit=20`, { admin });
    if ((persisted.messages || []).length < 4) {
      throw new Error('Expected create + edit chat turns persisted');
    }

    await reloadThreadChatWorkspace(page, deepLink, afterRecord.id);
    await guard.capture(page, shots.reload, { requireChatWorkspace: true });

    const screenshotMeta = guard.paths.map((p) => ({
      path: p,
      basename: path.basename(p),
      md5: crypto.createHash('md5').update(fs.readFileSync(p)).digest('hex'),
    }));

    const metadata = {
      tested_sha: testedSha,
      model: MODEL,
      thread_name: threadName,
      thread_id: threadId,
      quoted_canvas_id: canvasId,
      canvas_revision_id: afterRecord.id,
      edit_title: afterRecord.title,
      evidence_rows_before: claimsBefore,
      evidence_rows_after: claimsAfter,
      prompts: { create: createPrompt, update: updatePrompt },
      screenshots: screenshotMeta,
      captured_at: new Date().toISOString(),
    };

    const metaPath = path.join(ART_DIR, 'canvas-supersede-ui-metadata.json');
    fs.writeFileSync(metaPath, `${JSON.stringify(metadata, null, 2)}\n`);
    console.log(JSON.stringify(metadata, null, 2));
  } finally {
    await context.close();
    await browser.close();
  }
}

main().catch((err) => {
  console.error('[canvas-supersede-ui-proof] FAILED:', err);
  process.exit(1);
});