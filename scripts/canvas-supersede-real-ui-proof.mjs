#!/usr/bin/env node
/**
 * Single-thread real UI proof: chat-created canvas → copy link → plain-language update → same id.
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

async function openCanvasInWorkspace(page, admin, threadId) {
  const deadline = Date.now() + 600_000;
  while (Date.now() < deadline) {
    const list = await apiJson(`${DEFAULT_API_URL}/api/threads/${threadId}/canvases?current_only=true`, { admin });
    const canvases = list.canvases || [];
    if (canvases.length > 0) {
      const { id: canvasId, title } = canvases[0];
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

async function listThreadCanvases(admin, threadId) {
  const list = await apiJson(`${DEFAULT_API_URL}/api/threads/${threadId}/canvases?current_only=false`, {
    admin,
  });
  return list.canvases || [];
}

async function waitForCanvasSupersede(page, admin, threadId, canvasId, beforeRecord) {
  const quoted = String(canvasId).toLowerCase();
  const deadline = Date.now() + 480_000;
  let afterRecord = beforeRecord;
  while (Date.now() < deadline) {
    const canvases = await listThreadCanvases(admin, threadId);
    const child = canvases.find(
      (row) => String(row.supersedes_id || '').toLowerCase() === quoted,
    );
    if (child) {
      afterRecord = await resolveCanvasRecord(admin, threadId, canvasId);
      return afterRecord;
    }
    afterRecord = await resolveCanvasRecord(admin, threadId, canvasId);
    if (afterRecord.id !== beforeRecord.id) {
      return afterRecord;
    }
    const beforeSections = (beforeRecord.spec?.sections || []).map((s) => s.title).join('|');
    const afterSections = (afterRecord.spec?.sections || []).map((s) => s.title).join('|');
    const addedSummarySection = /summary/i.test(afterSections) && !/summary/i.test(beforeSections);
    const beforeSummary = String(beforeRecord.summary || beforeRecord.spec?.summary || '');
    const afterSummary = String(afterRecord.summary || afterRecord.spec?.summary || '');
    const summaryChanged = afterSummary.length > beforeSummary.length + 12;
    if (addedSummarySection || summaryChanged) {
      return afterRecord;
    }
    await page.waitForTimeout(8000);
  }
  throw new Error('Canvas was not superseded after the update request');
}

function parseCanvasDeepLink(url) {
  const u = new URL(url);
  const threadId = u.searchParams.get('thread');
  const canvasId = u.searchParams.get('canvas');
  const projectId = u.searchParams.get('project');
  if (!threadId || !canvasId) throw new Error(`Clipboard is not a canvas deep link: ${url}`);
  return { threadId, canvasId, projectId, url: u.toString() };
}

function sectionTitles(spec) {
  const safe = spec && typeof spec === 'object' ? spec : {};
  return (safe.sections || []).map((section) => String(section.title || '').trim()).filter(Boolean);
}

/**
 * Full page load of the thread workspace. The SPA only parses canvas deep links when
 * both thread and canvas query params are present (see parseCanvasWorkspaceDeepLink).
 */
async function reloadThreadChatWorkspace(page, deepLinkUrl, lineageHeadCanvasId) {
  const reloadTarget = new URL(deepLinkUrl);
  reloadTarget.searchParams.set('canvas', lineageHeadCanvasId);
  const threadId = reloadTarget.searchParams.get('thread');
  if (!threadId) throw new Error('Deep link missing thread id');

  const deadline = Date.now() + 240_000;
  while (Date.now() < deadline) {
    await page.goto(reloadTarget.toString(), { waitUntil: 'domcontentloaded', timeout: 120_000 });
    await waitForNoLoadingShell(page);
    const onThread = new URL(page.url()).searchParams.get('thread') === threadId
      || await page.getByTestId('chat-message-list').isVisible().catch(() => false);
    if (onThread) {
      try {
        await waitForThreadWorkspaceReady(page);
        break;
      } catch {
        await page.waitForTimeout(2500);
        continue;
      }
    }
    await page.waitForTimeout(2500);
  }
  await page.getByTestId('chat-message-list').waitFor({ state: 'visible', timeout: 180_000 });
  await page.getByText('Research canvas', { exact: true }).nth(1).waitFor({
    state: 'visible',
    timeout: 180_000,
  });
}

async function openCanvasFromReloadedThread(page, canvasTitle) {
  const overviewTab = page.getByRole('tab', { name: /Overview/i }).first();
  if (await overviewTab.isVisible({ timeout: 10_000 }).catch(() => false)) {
    await overviewTab.click();
  }
  const card = page.getByRole('button', { name: new RegExp(canvasTitle, 'i') }).last();
  if (await card.isVisible({ timeout: 15_000 }).catch(() => false)) {
    await card.click();
    return;
  }
  await page.getByText(canvasTitle, { exact: false }).last().click();
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

async function resolveCanvasRecord(admin, threadId, canvasId) {
  return apiJson(
    `${DEFAULT_API_URL}/api/threads/${threadId}/canvases/${canvasId}?resolve_current=true`,
    { admin },
  );
}

function canvasSpecMetrics(spec) {
  const safe = spec && typeof spec === 'object' ? spec : {};
  let claimCount = 0;
  let sourceCount = 0;
  for (const section of safe.sections || []) {
    for (const block of section.blocks || []) {
      if (block.type === 'table' && Array.isArray(block.rows)) {
        claimCount += block.rows.length;
      }
      if (block.type === 'timeline' && Array.isArray(block.events)) {
        claimCount += block.events.length;
      }
      if (block.type === 'claim_pair') {
        claimCount += 2;
      }
      if ((block.type === 'source_list' || block.type === 'sources') && Array.isArray(block.citations)) {
        sourceCount += block.citations.length;
      }
    }
  }
  return { claimCount, sourceCount };
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
    composer: 'chat-quote-canvas-1440x900.png',
    assistant: 'chat-assistant-reply-1440x900.png',
    after: 'canvas-after-same-id-1440x900.png',
    reload: 'chat-after-reload-1440x900.png',
    reloadCanvas: 'canvas-after-reload-1440x900.png',
  };

  try {
    await page.goto(DEFAULT_BASE_URL, { waitUntil: 'domcontentloaded', timeout: 120_000 });
    const { threadName } = await ensureProjectAndThread(page);
    const uiThreadId = await configureThreadForProof(admin, threadName);
    await selectLlmModel(page, MODEL);
    await waitForThreadWorkspaceReady(page);

    const createPrompt =
      'Please create a brief research canvas about how OpenRouter lists free models. '
      + 'Include several distinct claims with web sources.';
    await chatRateLimitPause();
    await typeAndSendChat(page, createPrompt);
    await waitForAssistantAnswer(page);

    const { title: canvasTitleBefore, canvasId: expectedCanvasId } = await openCanvasInWorkspace(page, admin, uiThreadId);
    await guard.capture(page, shots.before, { requireChatWorkspace: true });

    const deepLink = await copyCanvasLinkFromTab(page);
    const { threadId, canvasId } = parseCanvasDeepLink(deepLink);
    if (canvasId !== expectedCanvasId) {
      throw new Error(`Copied canvas id ${canvasId} does not match ${expectedCanvasId}`);
    }

    const beforeRecord = await resolveCanvasRecord(admin, threadId, canvasId);
    const beforeMetrics = canvasSpecMetrics(beforeRecord.spec);
    if (beforeMetrics.claimCount < 2) {
      throw new Error(`Pre-update canvas needs at least 2 claims for preservation proof (got ${beforeMetrics.claimCount})`);
    }

    await focusChatComposer(page);
    const updatePrompt = `Add a short summary section at the top.\n\n${deepLink}`;
    const composer = page.getByTestId('chat-composer-input');
    await composer.click();
    await composer.fill(updatePrompt);
    await guard.capture(page, shots.composer, { requireChatWorkspace: true });

    await chatRateLimitPause();
    await page.keyboard.press('Enter');
    const assistantPreview = await waitForAssistantAnswer(page);
    if (!/Open trace|research canvas|Published/i.test(assistantPreview)) {
      throw new Error(`Assistant reply lacks visible chat content: ${assistantPreview.slice(0, 120)}`);
    }
    await guard.capture(page, shots.assistant, { requireChatWorkspace: true });

    let afterRecord = beforeRecord;
    try {
      afterRecord = await waitForCanvasSupersede(page, admin, threadId, canvasId, beforeRecord);
    } catch (err) {
      await focusChatComposer(page);
      await typeAndSendChat(
        page,
        `Please update and republish that canvas with a short summary at the top, keeping all existing claims.\n\n${deepLink}`,
      );
      await chatRateLimitPause();
      await waitForAssistantAnswer(page);
      afterRecord = await waitForCanvasSupersede(page, admin, threadId, canvasId, beforeRecord);
      if (afterRecord.id === beforeRecord.id) {
        throw err;
      }
    }

    await openCanvasInWorkspace(page, admin, threadId);
    await page.getByRole('heading', { name: afterRecord.title }).first().waitFor({ timeout: 120_000 });
    await guard.capture(page, shots.after, { requireChatWorkspace: true });

    const persisted = await apiJson(`${DEFAULT_API_URL}/api/threads/${threadId}/messages?limit=20`, { admin });
    const messageCount = Array.isArray(persisted.messages) ? persisted.messages.length : 0;
    if (messageCount < 4) {
      throw new Error(`Expected at least 4 persisted messages after update, got ${messageCount}`);
    }

    const beforeTitles = new Set(sectionTitles(beforeRecord.spec));
    const afterMetrics = canvasSpecMetrics(afterRecord.spec);
    const afterTitles = sectionTitles(afterRecord.spec);
    if (
      afterMetrics.claimCount < beforeMetrics.claimCount
      || afterMetrics.sourceCount < beforeMetrics.sourceCount
    ) {
      throw new Error(
        `Canvas content shrank after update (claims ${beforeMetrics.claimCount} -> ${afterMetrics.claimCount}, `
        + `sources ${beforeMetrics.sourceCount} -> ${afterMetrics.sourceCount})`,
      );
    }
    const addedSection = afterTitles.some((title) => !beforeTitles.has(title));
    if (!addedSection && afterTitles.length <= beforeTitles.size) {
      throw new Error(
        `Supersede did not add a section (before=${afterTitles.length} titles=${afterTitles.join('|')})`,
      );
    }

    const resolvedHeadId = afterRecord.id;
    await reloadThreadChatWorkspace(page, deepLink, resolvedHeadId);
    await page.getByText(updatePrompt.split('\n')[0], { exact: false }).first()
      .waitFor({ state: 'visible', timeout: 120_000 });
    await guard.capture(page, shots.reload, { requireChatWorkspace: true });

    await openCanvasFromReloadedThread(page, afterRecord.title);
    await page.getByRole('heading', { name: afterRecord.title }).first().waitFor({ timeout: 120_000 });
    await guard.capture(page, shots.reloadCanvas, { requireChatWorkspace: true });

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
      canvas_id_before_lineage_head: beforeRecord.id,
      canvas_id_after_lineage_head: resolvedHeadId,
      canvas_title_before_tab: canvasTitleBefore,
      canvas_title_after: afterRecord.title,
      deep_link: deepLink,
      claim_count_before: beforeMetrics.claimCount,
      claim_count_after: afterMetrics.claimCount,
      source_count_before: beforeMetrics.sourceCount,
      source_count_after: afterMetrics.sourceCount,
      prompts: { create: createPrompt, update: 'Add a short summary section at the top. (+ pasted link)' },
      assistant_preview: assistantPreview.slice(0, 500),
      persisted_message_count: messageCount,
      screenshots: screenshotMeta,
      captured_at: new Date().toISOString(),
    };

    if (threadId !== uiThreadId || threadId !== beforeRecord.thread_id) {
      throw new Error('Thread id mismatch in proof run');
    }

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