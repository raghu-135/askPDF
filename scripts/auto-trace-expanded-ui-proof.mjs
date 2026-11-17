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

async function openTraceOnLatestMessage(page) {
  const openTrace = page.getByRole('button', { name: /open trace|debug trace/i }).first();
  if (await openTrace.isVisible({ timeout: 15_000 }).catch(() => false)) {
    await openTrace.click();
    await page.waitForTimeout(1200);
    return true;
  }
  return false;
}

async function expandTimelineNode(page, label) {
  const summary = page.locator('.MuiAccordionSummary-root').filter({ hasText: label }).first();
  await summary.waitFor({ state: 'visible', timeout: 60_000 });
  const expanded = await summary.getAttribute('aria-expanded');
  if (expanded !== 'true') await summary.click();
  await page.waitForTimeout(600);
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
    await page.goto(DEFAULT_BASE_URL, { waitUntil: 'domcontentloaded', timeout: 120_000 });
    const { threadName } = await ensureProjectAndThread(page);
    const threadId = await threadIdFromName(admin, threadName);
    await selectLlmModel(page, MODEL);
    await uploadPdfThroughUi(page, PDF);
    await waitForComposerReady(page);

    await page
      .getByRole('button', { name: /tune|context \d/i })
      .first()
      .click({ timeout: 10_000 })
      .catch(async () => {
        await page.getByLabel(/Initial retrieval budget|Context packing/i).first().click();
      });
    await guard.capture(page, `proof-01-initial-retrieval-budget-thread-${String(threadId).slice(0, 8)}.png`, { requireChatWorkspace: true });
    await page.keyboard.press('Escape');

    await apiJson(`${DEFAULT_API_URL}/api/threads/${threadId}/settings`, {
      method: 'PATCH',
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
      'Compare the method in this PDF to standard attention. Use a delegated workflow if needed and be thorough.',
    );
    await waitForAssistantAnswer(page, 240_000);
    await guard.capture(page, `proof-02-chat-after-auto-thread-${String(threadId).slice(0, 8)}.png`);

    const limitBanner = page.getByText(/hit the thread limit/i);
    if (await limitBanner.isVisible({ timeout: 3000 }).catch(() => false)) {
      await guard.capture(page, `proof-03-auto-limit-message-thread-${String(threadId).slice(0, 8)}.png`);
    }

    if (await openTraceOnLatestMessage(page)) {
      await expandTimelineNode(page, 'Context loader');
      await expandDisclosure(page, 'Prefetch context');
      await guard.capture(page, `proof-04-context-loader-expanded-thread-${String(threadId).slice(0, 8)}.png`);
      await expandTimelineNode(page, 'Workflow delegate');
      await expandDisclosure(page, 'Delegated child');
      await guard.capture(page, `proof-05-workflow-delegate-expanded-thread-${String(threadId).slice(0, 8)}.png`);
      const openChild = page.getByRole('button', { name: /Open child run in Debug Trace/i }).first();
      if (await openChild.isVisible({ timeout: 5000 }).catch(() => false)) {
        await openChild.click();
        await page.waitForTimeout(1500);
        await expandTimelineNode(page, 'Planner');
        await expandDisclosure(page, 'Prompt');
        await guard.capture(page, `proof-06-child-trace-opened-thread-${String(threadId).slice(0, 8)}.png`);
      }
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