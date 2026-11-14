import { test, expect } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const fixture = path.join(__dirname, 'fixtures', 'chat-web-search-approval-panel.html');
const buildScript = path.join(__dirname, 'scripts', 'build-chat-web-approval-fixture.mjs');

test.beforeAll(() => {
  execFileSync(process.execPath, [buildScript], { cwd: path.join(__dirname, '..') });
});

async function assertCompactToolApprovalLayout(page) {
  const panel = page.getByTestId('chat-human-review-panel');
  const approve = page.getByRole('button', { name: /Approve once/i });
  const composer = page.getByTestId('chat-composer');
  const transcript = page.getByTestId('chat-transcript-frame');

  await expect(panel).toBeVisible();
  await expect(approve).toBeVisible();
  await expect(composer).toBeVisible();
  await expect(transcript).toBeVisible();
  await expect(panel).toContainText(/Search the web for:/i);

  const viewport = page.viewportSize();
  const approveBox = await approve.boundingBox();
  const composerBox = await composer.boundingBox();
  const transcriptBox = await transcript.boundingBox();
  expect(approveBox).not.toBeNull();
  expect(composerBox).not.toBeNull();
  expect(transcriptBox).not.toBeNull();
  expect(transcriptBox.height).toBeGreaterThan(80);
  expect(composerBox.y).toBeGreaterThan(approveBox.y);
  expect(composerBox.y + composerBox.height).toBeLessThanOrEqual((viewport?.height || 900) + 1);
}

const viewports = [
  { name: '1440x900', width: 1440, height: 900 },
  { name: '1024x768', width: 1024, height: 768 },
];

test.describe('Chat web search approval compact card', () => {
  for (const viewport of viewports) {
    test(viewport.name, async ({ page }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.goto(`file://${fixture}`);
      await assertCompactToolApprovalLayout(page);
    });
  }
});