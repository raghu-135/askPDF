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

async function assertApprovalButtonsFullyVisible(page) {
  const panel = page.getByTestId('chat-human-review-panel');
  const footer = page.getByTestId('review-decision-footer');
  const approve = page.getByRole('button', { name: /Approve once/i });
  const deny = page.getByRole('button', { name: /Skip this tool/i });

  await expect(panel).toBeVisible();
  await expect(footer).toBeVisible();
  await expect(approve).toBeVisible();
  await expect(deny).toBeVisible();

  const viewport = page.viewportSize();
  const approveBox = await approve.boundingBox();
  const denyBox = await deny.boundingBox();
  const footerBox = await footer.boundingBox();
  expect(approveBox).not.toBeNull();
  expect(denyBox).not.toBeNull();
  expect(footerBox).not.toBeNull();

  for (const box of [approveBox, denyBox]) {
    expect(box.y + box.height).toBeLessThanOrEqual((viewport?.height || 800) + 0.5);
    expect(box.y).toBeGreaterThanOrEqual(0);
  }

  const overlap = await page.evaluate(() => {
    const scroll = document.querySelector('[data-testid="chat-human-review-panel"]');
    const foot = document.querySelector('[data-testid="review-decision-footer"]');
    if (!scroll || !foot) return null;
    return scroll.getBoundingClientRect().bottom - foot.getBoundingClientRect().top;
  });
  expect(overlap).not.toBeNull();
  expect(overlap).toBeLessThanOrEqual(0.5);
}

const viewports = [
  { name: '1280x800', width: 1280, height: 800 },
  { name: '1024x768', width: 1024, height: 768 },
];

test.describe('Chat web search approval panel layout', () => {
  for (const viewport of viewports) {
    test(viewport.name, async ({ page }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.goto(`file://${fixture}`);
      await assertApprovalButtonsFullyVisible(page);
    });
  }
});