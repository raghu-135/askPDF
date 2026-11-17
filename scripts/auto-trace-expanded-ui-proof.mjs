#!/usr/bin/env node
/** Expanded Debug Trace + Auto limits + limit-hit proof (1440x900). */
import path from 'node:path';
import { createRequire } from 'node:module';
import {
  DEFAULT_ART_DIR,
  DEFAULT_API_URL,
  DEFAULT_BASE_URL,
  REAL_UI_VIEWPORT,
  ScreenshotGuard,
  apiJson,
  chatRateLimitPause,
  ensureProjectAndThread,
  loadEnvAdmin,
  selectLlmModel,
  typeAndSendChat,
  uploadPdfThroughUi,
  waitForAssistantAnswer,
  waitForComposerReady,
} from './cloud-agent-e2e-lib.mjs';

const requireFromFrontend = createRequire(path.join(process.cwd(), 'frontend/package.json'));
const { chromium } = requireFromFrontend('@playwright/test');
const MODEL = process.env.ASKPDF_E2E_LLM_MODEL || 'inclusionai/ling-3.2-flash:free';
const PDF = process.env.ASKPDF_E2E_PDF || path.join(process.cwd(), 'rag_service/tests/01030000000000.pdf');

async function threadIdFromName(admin, threadName) {
  const threads = await apiJson(`${DEFAULT_API_URL}/api/threads`, { admin });
  const row = (threads.threads || []).find((t) => t.name === threadName);
  return row?.id;
}

async function resolveThreadId(page, admin, ensured) {
  const fromUrl = page.url().match(/\/threads\/([^/?]+)/)?.[1];
  if (fromUrl) return fromUrl;
  if (ensured.threadName) return threadIdFromName(admin, ensured.threadName);
  return null;
}

async function showThreadIdInLocationBar(page, threadId) {
  await page.evaluate((id) => {
    const url = new URL(window.location.href);
    url.searchParams.set('thread', id);
    window.history.replaceState({}, '', url.toString());
  }, threadId);
}

async function openDebugTraceWorkspace(page) {
  const openTrace = page.getByText(/Open trace ·/i).last();
  if (!(await openTrace.isVisible({ timeout: 30_000 }).catch(() => false))) {
    return false;
  }
  await openTrace.scrollIntoViewIfNeeded();
  await openTrace.click();
  const debugTab = page.getByRole('tab', { name: /^Debug Trace$/i });
  await debugTab.waitFor({ state: 'visible', timeout: 30_000 });
  await debugTab.click();
  await page.getByLabel('Open traces').waitFor({ state: 'visible', timeout: 120_000 });
  await page.waitForTimeout(1000);
  return true;
}

async function expandTimelineNode(page, label) {
  const needle = String(label).replace(/\s+/g, '_').toLowerCase();
  const rowLabel = page
    .getByText(new RegExp(`^${needle}$`, 'i'))
    .or(page.getByText(new RegExp(label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i')))
    .first();
  await rowLabel.scrollIntoViewIfNeeded({ timeout: 60_000 });
  await rowLabel.waitFor({ state: 'visible', timeout: 120_000 });
  await rowLabel.click();
  await page.waitForTimeout(700);
}

async function expandDisclosure(page, label) {
  const btn = page.getByRole('button', { name: new RegExp(label, 'i') }).first();
  if (await btn.isVisible({ timeout: 5000 }).catch(() => false)) {
    await btn.click();
    await page.waitForTimeout(400);
  }
}

async function main() {
  const admin = loadEnvAdmin();
  const guard = new ScreenshotGuard(DEFAULT_ART_DIR);
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: REAL_UI_VIEWPORT });

  try {
    const ensured = await ensureProjectAndThread(page);
    const threadId = await resolveThreadId(page, admin, ensured);
    if (!threadId) throw new Error('Could not resolve thread id');

    await selectLlmModel(page, MODEL);
    await uploadPdfThroughUi(page, PDF);
    await waitForComposerReady(page);
    await showThreadIdInLocationBar(page, threadId);

    await page.getByRole('button', { name: /^Context / }).click({ timeout: 15_000 });
    await page.getByText('Initial retrieval budget', { exact: false }).waitFor({ state: 'visible', timeout: 15_000 });
    await guard.capture(page, `proof-01-initial-retrieval-budget-thread-${String(threadId).slice(0, 8)}.png`, {
      requireChatWorkspace: false,
    });
    await page.keyboard.press('Escape');
    await page.waitForTimeout(500);

    await apiJson(`${DEFAULT_API_URL}/api/threads/${threadId}/settings`, {
      method: 'PUT',
      admin,
      body: {
        agent_workflow: { workflow_id: 'auto_rag_agent' },
        auto_limits: {
          max_total_visits: 4,
          max_interrupts_per_run: 1,
          max_delegates_per_turn: 1,
          require_plan_approval_on_complex: false,
          require_plan_approval_all: false,
        },
      },
    });

    await chatRateLimitPause();
    await typeAndSendChat(
      page,
      'Summarize this PDF in two sentences, then delegate a focused follow-up to the standard RAG workflow for one extra detail (arXiv id and date).',
    );
    await waitForAssistantAnswer(page, 300_000);
    await showThreadIdInLocationBar(page, threadId);
    await guard.capture(page, `proof-02-chat-after-auto-thread-${String(threadId).slice(0, 8)}.png`);

    const limitBanner = page.getByText(/hit the thread limit/i);
    if (await limitBanner.isVisible({ timeout: 5000 }).catch(() => false)) {
      await guard.capture(page, `proof-03-auto-limit-message-thread-${String(threadId).slice(0, 8)}.png`);
    }

    if (await openDebugTraceWorkspace(page)) {
      await showThreadIdInLocationBar(page, threadId);
      await page.waitForFunction(
        () => /Context Loader|context_loader|Workflow Delegate|workflow_delegate/i.test(document.body.innerText || ''),
        undefined,
        { timeout: 180_000 },
      );
      try {
        await expandTimelineNode(page, 'context_loader');
        await expandDisclosure(page, 'Prefetch context');
        await expandDisclosure(page, 'Raw JSON');
      } catch (error) {
        console.warn('[proof] Context Loader expand:', error?.message || error);
      }
      await guard.capture(page, `proof-04-context-loader-expanded-thread-${String(threadId).slice(0, 8)}.png`, {
        requireChatWorkspace: false,
      });
      try {
        await expandTimelineNode(page, 'auto_orchestrator');
        await expandDisclosure(page, 'Prompt');
      } catch (error) {
        console.warn('[proof] Prompt expand:', error?.message || error);
      }
      await guard.capture(page, `proof-04b-prompts-expanded-thread-${String(threadId).slice(0, 8)}.png`, {
        requireChatWorkspace: false,
      });
      try {
        await expandTimelineNode(page, 'workflow_delegate');
        await expandDisclosure(page, 'Delegated child run');
      } catch (error) {
        console.warn('[proof] Workflow Delegate expand:', error?.message || error);
      }
      await guard.capture(page, `proof-05-workflow-delegate-expanded-thread-${String(threadId).slice(0, 8)}.png`, {
        requireChatWorkspace: false,
      });
      const openChild = page.getByRole('button', { name: /Open child run in Debug Trace/i }).first();
      if (await openChild.isVisible({ timeout: 8000 }).catch(() => false)) {
        await openChild.scrollIntoViewIfNeeded();
        await openChild.click();
        await page.waitForTimeout(2000);
        await page.getByLabel('Open traces').waitFor({ state: 'visible', timeout: 60_000 }).catch(() => {});
        try {
          await expandTimelineNode(page, 'planner');
          await expandDisclosure(page, 'Prompt');
        } catch (error) {
          console.warn('[proof] Child trace expand:', error?.message || error);
        }
        await page.mouse.wheel(0, 400);
        await page.waitForTimeout(400);
        await guard.capture(page, `proof-06-child-trace-opened-thread-${String(threadId).slice(0, 8)}.png`, {
          requireChatWorkspace: false,
        });
      }
    }

    if (!(await limitBanner.isVisible({ timeout: 1000 }).catch(() => false))) {
      await chatRateLimitPause();
      await typeAndSendChat(page, 'Keep replanning and delegating on Auto until you cannot take another workflow step.');
      await waitForAssistantAnswer(page, 300_000).catch(() => null);
    }
    const limitText = page.getByText(/hit the thread limit/i);
    if (await limitText.isVisible({ timeout: 8000 }).catch(() => false)) {
      await showThreadIdInLocationBar(page, threadId);
      await guard.capture(page, `proof-03-auto-limit-message-thread-${String(threadId).slice(0, 8)}.png`, {
        requireChatWorkspace: true,
      });
    }

    if (!guard.paths.length) {
      throw new Error('No proof screenshots captured');
    }
    console.log('[proof] thread_id', threadId);
    console.log('[proof] artifacts', guard.paths);
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error('[auto-trace-expanded-ui-proof] FAILED', err);
  process.exit(1);
});