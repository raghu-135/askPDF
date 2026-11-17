/** Shared Node helpers for cloud-agent e2e scripts. */
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = path.resolve(__dirname, '..');

export const DEFAULT_ART_DIR = process.env.ASKPDF_E2E_ARTIFACTS || '/opt/cursor/artifacts';
export const DEFAULT_BASE_URL = process.env.ASKPDF_E2E_BASE_URL || 'http://127.0.0.1:3000';
export const DEFAULT_API_URL = process.env.ASKPDF_E2E_API_URL || 'http://127.0.0.1:8000';

export function loadEnvAdmin() {
  const raw = fs.readFileSync(path.join(REPO_ROOT, '.env'), 'utf8');
  const admin = raw.match(/^ASKPDF_ADMIN_TOKEN=(.+)$/m)?.[1]?.trim();
  if (!admin) throw new Error('ASKPDF_ADMIN_TOKEN missing in .env');
  return admin;
}

export async function apiJson(url, { method = 'GET', body, admin } = {}) {
  const res = await fetch(url, {
    method,
    headers: {
      Authorization: `Bearer ${admin}`,
      ...(body ? { 'Content-Type': 'application/json' } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${method} ${url} failed: ${res.status} ${text.slice(0, 400)}`);
  }
  return res.json();
}

export async function waitForFileReady(threadId, fileHash, admin, apiUrl = DEFAULT_API_URL) {
  for (let i = 0; i < 240; i += 1) {
    const status = await apiJson(`${apiUrl}/api/threads/${threadId}/files/${fileHash}/status`, { admin });
    const indexing = status?.indexing?.status;
    const projection = status?.document_processing?.projection?.status;
    if (indexing === 'completed' && projection === 'completed') return;
    await new Promise((r) => setTimeout(r, 3000));
  }
  throw new Error(`file ${fileHash} not indexed in time`);
}

export function defaultProjectId(projectsPayload) {
  const rows = Array.isArray(projectsPayload) ? projectsPayload : projectsPayload?.projects || [];
  return rows[0]?.id;
}

export const REAL_UI_VIEWPORT = { width: 1440, height: 900 };

const MIN_CHAT_BACKOFF_MS = Number(process.env.ASKPDF_E2E_CHAT_BACKOFF_MS || 4000);

const LOADING_SCREEN_PATTERNS = [
  /Loading thread/i,
  /Loading project details/i,
];

function md5File(filePath) {
  const data = fs.readFileSync(filePath);
  return crypto.createHash('md5').update(data).digest('hex');
}

/** Tracks PNG paths and fails if any two captures share an MD5 digest. */
export class ScreenshotGuard {
  /** @param {string} artDir */
  constructor(artDir) {
    this.artDir = artDir;
    this.byHash = new Map();
    this.paths = [];
    fs.mkdirSync(artDir, { recursive: true });
  }

  /**
   * @param {import('playwright').Page} page
   * @param {string} filename
   * @param {{ requireChatWorkspace?: boolean, allowLoadingShell?: boolean }} [options]
   */
  async capture(page, filename, options = {}) {
    const { requireChatWorkspace = false, allowLoadingShell = false } = options;
    if (!allowLoadingShell) {
      await assertPageReadyForScreenshot(page, { requireChatWorkspace });
    }
    const out = path.join(this.artDir, filename);
    await page.screenshot({ path: out, fullPage: false });
    this.register(out);
    return out;
  }

  /** @param {string} filePath */
  register(filePath) {
    const hash = md5File(filePath);
    const prior = this.byHash.get(hash);
    if (prior) {
      throw new Error(
        `Duplicate screenshot: ${path.basename(filePath)} is byte-identical to ${path.basename(prior)} (md5=${hash})`,
      );
    }
    this.byHash.set(hash, filePath);
    this.paths.push(filePath);
    return hash;
  }
}

/**
 * Prove duplicate detection without a browser (copies one PNG twice).
 * @param {string} artDir
 */
export function proveScreenshotGuardFailsOnDuplicate(artDir) {
  fs.mkdirSync(artDir, { recursive: true });
  const seed = path.join(artDir, '_guard-seed.png');
  if (!fs.existsSync(seed)) {
    const png = Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
      'base64',
    );
    fs.writeFileSync(seed, png);
  }
  const copy = path.join(artDir, '_guard-copy.png');
  fs.copyFileSync(seed, copy);
  const guard = new ScreenshotGuard(artDir);
  guard.register(seed);
  try {
    guard.register(copy);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (!message.includes('Duplicate screenshot')) {
      throw new Error(`Expected duplicate guard failure, got: ${message}`);
    }
    return message;
  }
  throw new Error('Screenshot guard did not reject duplicate files');
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * @param {import('playwright').Page} page
 * @param {{ timeout?: number }} [options]
 */
export async function waitForNoLoadingShell(page, { timeout = 180_000 } = {}) {
  for (const pattern of LOADING_SCREEN_PATTERNS) {
    const label = page.getByText(pattern).first();
    if (await label.isVisible({ timeout: 1500 }).catch(() => false)) {
      await label.waitFor({ state: 'hidden', timeout });
    }
  }
  await page.waitForFunction(
    (patterns) => {
      const text = document.body?.innerText || '';
      return !patterns.some((p) => new RegExp(p, 'i').test(text));
    },
    LOADING_SCREEN_PATTERNS.map((p) => p.source),
    { timeout },
  );
}

/**
 * @param {import('playwright').Page} page
 * @param {{ requireChatWorkspace?: boolean, timeout?: number }} [options]
 */
export async function assertPageReadyForScreenshot(page, { requireChatWorkspace = false, timeout = 180_000 } = {}) {
  await waitForNoLoadingShell(page, { timeout });
  if (requireChatWorkspace) {
    await assertChatWorkspaceVisible(page);
  }
  for (const pattern of LOADING_SCREEN_PATTERNS) {
    if (await page.getByText(pattern).first().isVisible().catch(() => false)) {
      throw new Error(`Screenshot blocked: visible loading state matching ${pattern}`);
    }
  }
}

/** Thread shell + chat composer/message list ready (post thread create or navigation). */
export async function waitForThreadWorkspaceReady(page) {
  await waitForNoLoadingShell(page);
  await assertChatWorkspaceVisible(page);
}

/**
 * @param {import('playwright').Page} page
 */
export async function assertChatWorkspaceVisible(page) {
  const llm = page.getByRole('combobox', { name: /Select LLM/i }).first();
  await llm.waitFor({ state: 'visible', timeout: 120_000 });
  const composer = page.getByTestId('chat-composer-input');
  await composer.waitFor({ state: 'visible', timeout: 120_000 });
  const transcript = page.getByTestId('chat-message-list');
  await transcript.waitFor({ state: 'visible', timeout: 120_000 });
}

/**
 * @param {import('playwright').Page} page
 * @param {string} [projectLabel]
 */
export async function bootstrapProjectThreadViaApi(page, projectLabel = `ui-proof-${Date.now()}`) {
  const admin = loadEnvAdmin();
  const project = await apiJson(`${DEFAULT_API_URL}/api/projects`, {
    method: 'POST',
    admin,
    body: {
      name: projectLabel,
      description: 'cloud-agent delegate UI proof',
      embedding_model: 'BAAI/bge-m3',
    },
  });
  const threadName = `thread-${Date.now()}`;
  const thread = await apiJson(`${DEFAULT_API_URL}/api/projects/${project.id}/threads`, {
    method: 'POST',
    admin,
    body: { name: threadName },
  });
  const threadId = String(thread.id);
  const projectId = String(project.id);
  await page.addInitScript(
    ({ resumeThreadId }) => {
      const key = 'askpdf.workspace.resume.v1';
      let existing = {};
      try {
        existing = JSON.parse(window.localStorage.getItem(key) || '{}');
      } catch {
        existing = {};
      }
      existing[`thread:${resumeThreadId}`] = {
        tabId: 'documents-tab',
        documentId: null,
        traceId: null,
        canvasId: null,
      };
      window.localStorage.setItem(key, JSON.stringify(existing));
    },
    { resumeThreadId: threadId },
  );
  const openUrl = new URL(DEFAULT_BASE_URL);
  openUrl.searchParams.set('project', projectId);
  openUrl.searchParams.set('thread', threadId);
  await page.goto(openUrl.toString(), { waitUntil: 'domcontentloaded', timeout: 120_000 });
  await waitForThreadWorkspaceReady(page);
  return {
    projectId,
    threadId,
    threadName,
    projectLabel,
  };
}

export async function ensureProjectAndThread(page, projectLabel = `ui-proof-${Date.now()}`) {
  await page.goto(DEFAULT_BASE_URL, { waitUntil: 'domcontentloaded', timeout: 120_000 });
  await page.waitForTimeout(1200);

  const llmCombo = page.getByRole('combobox', { name: /Select LLM/i }).first();
  if (await llmCombo.isVisible({ timeout: 3000 }).catch(() => false)) {
    await waitForThreadWorkspaceReady(page);
    return { projectLabel, reused: true };
  }

  const createInExisting = page.getByRole('button', { name: /Create thread in /i }).first();
  if (!(await createInExisting.isVisible({ timeout: 4000 }).catch(() => false))) {
    const createProject = page
      .getByRole('button', { name: /Create project/i })
      .or(page.getByRole('button', { name: /Create Project/i }))
      .first();
    await createProject.waitFor({ timeout: 60_000 });
    await createProject.evaluate((button) => {
      button.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    const createProjectDialog = page.getByRole('dialog').first();
    await createProjectDialog.waitFor({ state: 'visible', timeout: 60_000 });
    await createProjectDialog.getByLabel('Project Name').fill(projectLabel);
    const embeddingSelect = createProjectDialog
      .getByRole('combobox', { name: /Embedding Model/i })
      .or(createProjectDialog.getByLabel('Embedding Model'))
      .first();
    await embeddingSelect.click();
    await page.getByRole('option', { name: /BAAI\/bge-m3/i }).first().click();
    const confirmCreateProject = createProjectDialog.getByRole('button', { name: /^Create Project$/i });
    for (let attempt = 0; attempt < 120; attempt += 1) {
      if (await confirmCreateProject.isEnabled().catch(() => false)) break;
      await page.waitForTimeout(1000);
    }
    await confirmCreateProject.evaluate((button) => {
      button.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await page.waitForTimeout(1200);
  } else {
    const match = await createInExisting.innerText();
    const fromButton = match.replace(/^Create thread in\s+/i, '').trim();
    if (fromButton) projectLabel = fromButton;
  }

  const threadName = `thread-${Date.now()}`;
  const createThreadBtn = page
    .getByRole('button', { name: new RegExp(`Create thread in ${escapeRegExp(projectLabel)}`, 'i') })
    .or(page.getByRole('button', { name: /Create Thread/i }))
    .first();
  await createThreadBtn.waitFor({ timeout: 60_000 });
  await createThreadBtn.click();
  await page.getByRole('textbox').first().fill(threadName);
  await page.getByRole('button', { name: /^Create$/i }).click();
  await llmCombo.waitFor({ timeout: 60_000 });
  await waitForThreadWorkspaceReady(page);
  return { projectLabel, threadName, reused: false };
}

/**
 * @param {import('playwright').Page} page
 */
/** Wait until document indexing finishes (composer may stay disabled until an LLM is chosen). */
export async function waitForSourcesIndexed(page) {
  const indexing = page.getByPlaceholder(/Indexing your sources/i);
  if (await indexing.isVisible({ timeout: 5000 }).catch(() => false)) {
    await indexing.waitFor({ state: 'hidden', timeout: 600_000 });
  }
  const blocked = page.getByPlaceholder(/Blocked: document indexing/i);
  if (await blocked.isVisible({ timeout: 1000 }).catch(() => false)) {
    throw new Error('Document indexing blocked in UI');
  }
}

export async function waitForComposerReady(page) {
  const checking = page.getByPlaceholder(/Checking LLM model/i);
  if (await checking.isVisible({ timeout: 2000 }).catch(() => false)) {
    await checking.waitFor({ state: 'hidden', timeout: 180_000 });
  }
  await waitForSourcesIndexed(page);
  const composer = page.getByTestId('chat-composer-input');
  await composer.waitFor({ state: 'visible', timeout: 180_000 });
  const noModel = page.getByPlaceholder(/Select LLM model/i);
  if (await noModel.isVisible({ timeout: 500 }).catch(() => false)) {
    throw new Error('Composer still locked: select an LLM model before waitForComposerReady');
  }
  await page.waitForFunction(
    () => {
      const el = document.querySelector('[data-testid="chat-composer-input"]');
      return el instanceof HTMLTextAreaElement && !el.disabled;
    },
    undefined,
    { timeout: 300_000 },
  );
}

/**
 * @param {import('playwright').Page} page
 * @param {string} model
 */
export async function selectLlmModel(page, model) {
  const combo = page.getByRole('combobox', { name: /Select LLM/i }).first();
  await combo.click();
  const short = model.includes('/') ? model.split('/').slice(-1)[0] : model;
  await combo.fill(short);
  await page.waitForTimeout(400);
  const option = page.getByRole('option', { name: new RegExp(escapeRegExp(model), 'i') }).first();
  if (await option.count()) {
    await option.click();
  } else {
    const loose = page.getByRole('option', { name: new RegExp(escapeRegExp(short), 'i') }).first();
    if (await loose.count()) {
      await loose.click();
    } else {
      await page.keyboard.press('Escape');
      throw new Error(`LLM model not found in UI: ${model}`);
    }
  }
  await waitForComposerReady(page);
}

/**
 * @param {import('playwright').Page} page
 * @param {string} pdfPath
 */
export async function uploadPdfThroughUi(page, pdfPath) {
  const base = path.basename(pdfPath);
  const input = page.getByTestId('pdf-upload-input');
  await input.waitFor({ state: 'attached', timeout: 60_000 });
  await input.setInputFiles(pdfPath);
  const docTab = page.getByRole('tab', { name: new RegExp(escapeRegExp(base.replace(/\.pdf$/i, '')), 'i') }).first();
  await docTab.waitFor({ state: 'visible', timeout: 300_000 });
  await waitForSourcesIndexed(page);
  return base;
}

/**
 * @param {import('playwright').Page} page
 * @param {string} text
 */
export async function typeAndSendChat(page, text) {
  const composer = page.getByTestId('chat-composer-input');
  await composer.click();
  await composer.fill(text);
  await page.keyboard.press('Enter');
}

/**
 * @param {import('playwright').Page} page
 */
export async function waitForAssistantAnswer(page, options = {}) {
  const mustMatch = options.mustMatch;
  const composer = page.getByTestId('chat-composer-input');
  await composer.waitFor({ timeout: 180_000 });
  const sending = page.getByPlaceholder(/Sending/i);
  const sendingTimeout = Number(options.sendingTimeoutMs)
    || (mustMatch ? 1_200_000 : 300_000);
  if (await sending.isVisible({ timeout: 1000 }).catch(() => false)) {
    await sending.waitFor({ state: 'hidden', timeout: sendingTimeout });
  }
  await page.waitForFunction(
    () => {
      const el = document.querySelector('[data-testid="chat-composer-input"]');
      return el instanceof HTMLTextAreaElement && !el.disabled;
    },
    undefined,
    { timeout: 600_000 },
  );
  const answer = page
    .getByTestId('chat-message-list')
    .locator('li, .MuiListItem-root')
    .filter({ hasText: /Open trace/i })
    .last();
  await answer.waitFor({ state: 'visible', timeout: 300_000 });
  await answer.scrollIntoViewIfNeeded();
  const body = await answer.innerText();
  if (body.length < 20) {
    throw new Error('Assistant answer too short for UI proof');
  }
  if (mustMatch instanceof RegExp && !mustMatch.test(body)) {
    throw new Error(`Assistant answer did not match required pattern: ${body.slice(0, 280)}`);
  }
  return body;
}

/**
 * @param {import('playwright').Page} page
 * @param {string} fileBaseName
 */
export async function openUploadedDocumentTab(page, fileBaseName) {
  const tab = page.getByRole('tab', { name: new RegExp(escapeRegExp(fileBaseName.replace(/\.pdf$/i, '')), 'i') }).first();
  if (await tab.isVisible({ timeout: 8000 }).catch(() => false)) {
    await tab.click();
    await page.waitForTimeout(800);
    return true;
  }
  const documentsTab = page.getByRole('tab', { name: /Documents/i }).first();
  if (await documentsTab.isVisible({ timeout: 3000 }).catch(() => false)) {
    await documentsTab.click();
    await page.waitForTimeout(800);
    return true;
  }
  throw new Error(`Document tab for ${fileBaseName} not visible`);
}

export async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Backoff between chat turns to stay under provider RPM limits. */
export async function chatRateLimitPause() {
  await sleep(MIN_CHAT_BACKOFF_MS);
}

/** Cycle the chat-header browser workbench control until Ask mode is active. */
export async function setBrowserAskMode(page) {
  await page.evaluate(() => {
    localStorage.setItem('last_browser_access_mode', 'ask');
    window.dispatchEvent(new CustomEvent('askpdf:browser-access-mode', { detail: 'ask' }));
  });
  const askBtn = page.getByRole('button', { name: /^Ask before the agent uses the workbench browser$/i });
  if (await askBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
    return;
  }
  const browserToggle = page
    .getByRole('button', {
      name: /Browser agent off|Browser agent on|Ask before the agent uses the workbench browser/i,
    })
    .first();
  await browserToggle.waitFor({ state: 'visible', timeout: 60_000 });
  for (let i = 0; i < 8; i += 1) {
    if (await askBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
      return;
    }
    const label = ((await browserToggle.textContent()) || '').trim();
    if (/browser agent off/i.test(label)) {
      await browserToggle.click();
      await page.waitForTimeout(500);
      continue;
    }
    if (/browser agent on/i.test(label)) {
      await browserToggle.click();
      await page.waitForTimeout(500);
      continue;
    }
    await browserToggle.click();
    await page.waitForTimeout(500);
  }
  if (await askBtn.isVisible({ timeout: 2000 }).catch(() => false)) return;
  throw new Error('Could not set browser access mode to Ask');
}

/** @param {import('playwright').Page} page */
export function clarificationPanel(page) {
  return page
    .locator('div')
    .filter({ has: page.getByText('I need a bit more clarification') })
    .filter({ has: page.getByRole('button', { name: /Something else/i }) })
    .last();
}

/** @param {import('playwright').Page} page */
export async function waitClarificationCard(page) {
  await page.getByText('I need a bit more clarification').waitFor({ timeout: 600_000 });
  await page.getByRole('button', { name: /Something else/i }).waitFor({ timeout: 60_000 });
}

/** @param {import('playwright').Page} page */
export async function waitOutOfSending(page) {
  const sending = page.getByPlaceholder(/Sending/i);
  if (await sending.isVisible({ timeout: 120_000 }).catch(() => false)) {
    await sending.waitFor({ state: 'hidden', timeout: 900_000 });
  }
}

/** @param {import('playwright').Page} page */
export async function chatTranscriptText(page) {
  return (await page.getByTestId('chat-message-list').innerText()).trim();
}

/** @param {string} text */
export function assertNotScopeDecline(text) {
  if (/attached documents do not address/i.test(text)) {
    throw new Error(`Got scope-decline answer: ${text.slice(0, 320)}`);
  }
}

/** @param {string} text */
export function assertGroundedPdfAnswer(text, { extraPattern } = {}) {
  assertNotScopeDecline(text);
  const pattern = extraPattern || /(MORAN|DINER|encoder|architecture|layer|attention|rectification)/i;
  if (!pattern.test(text)) {
    throw new Error(`Answer does not look PDF-grounded: ${text.slice(-420)}`);
  }
}

/**
 * After a repeat ambiguous send, accept either a new clarification card or a grounded answer.
 * @param {import('playwright').Page} page
 */
export async function waitRepeatClarifyOrGrounded(page, timeoutMs = 600_000) {
  const card = page.getByText('I need a bit more clarification');
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await card.isVisible().catch(() => false)) {
      return 'clarification';
    }
    const text = await chatTranscriptText(page);
    if (!/Sending/i.test(text)) {
      assertNotScopeDecline(text);
      if (/(MORAN|encoder|architecture|layer|attention|compare)/i.test(text)) {
        return 'answered';
      }
    }
    await sleep(2000);
  }
  throw new Error('Timed out waiting for clarification card or grounded repeat answer');
}