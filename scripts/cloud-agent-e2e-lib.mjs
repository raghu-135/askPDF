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
   */
  async capture(page, filename) {
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
export async function ensureProjectAndThread(page, projectLabel = `ui-proof-${Date.now()}`) {
  await page.goto(DEFAULT_BASE_URL, { waitUntil: 'domcontentloaded', timeout: 120_000 });
  await page.waitForTimeout(1200);

  const llmCombo = page.getByRole('combobox', { name: /Select LLM/i }).first();
  if (await llmCombo.isVisible({ timeout: 3000 }).catch(() => false)) {
    return { projectLabel, reused: true };
  }

  const createInExisting = page.getByRole('button', { name: /Create thread in /i }).first();
  if (!(await createInExisting.isVisible({ timeout: 4000 }).catch(() => false))) {
    const createProject = page.getByRole('button', { name: /Create Project/i });
    await createProject.waitFor({ timeout: 60_000 });
    await createProject.click();
    await page.getByRole('textbox').first().fill(projectLabel);
    await page.getByRole('button', { name: /^Create Project$/i }).click();
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
  await assertChatWorkspaceVisible(page);
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
export async function waitForAssistantAnswer(page) {
  const composer = page.getByTestId('chat-composer-input');
  await composer.waitFor({ timeout: 180_000 });
  const sending = page.getByPlaceholder(/Sending/i);
  if (await sending.isVisible({ timeout: 1000 }).catch(() => false)) {
    await sending.waitFor({ state: 'hidden', timeout: 300_000 });
  }
  await page.waitForFunction(
    () => {
      const el = document.querySelector('[data-testid="chat-composer-input"]');
      return el instanceof HTMLTextAreaElement && !el.disabled;
    },
    undefined,
    { timeout: 300_000 },
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