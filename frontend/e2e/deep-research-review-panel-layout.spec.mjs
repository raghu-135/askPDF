import { test, expect } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function assertFooterButtonsInViewportAndClickable(page) {
  const finish = page.getByRole('button', { name: 'Finish with this result' });
  const continueBtn = page.getByRole('button', { name: 'Continue with guidance' });
  await expect(finish).toBeVisible();
  await expect(continueBtn).toBeVisible();

  const finishBox = await finish.boundingBox();
  const continueBox = await continueBtn.boundingBox();
  const viewport = page.viewportSize();
  expect(finishBox).not.toBeNull();
  expect(continueBox).not.toBeNull();
  expect(viewport).not.toBeNull();

  expect(finishBox.y + finishBox.height).toBeLessThanOrEqual(viewport.height);
  expect(continueBox.y + continueBox.height).toBeLessThanOrEqual(viewport.height);
  expect(finishBox.y).toBeGreaterThanOrEqual(0);
  expect(continueBox.y).toBeGreaterThanOrEqual(0);

  for (const [label, box] of [['Finish with this result', finishBox], ['Continue with guidance', continueBox]]) {
    const centerX = box.x + box.width / 2;
    const centerY = box.y + box.height / 2;
    const hitLabel = await page.evaluate(({ x, y, expected }) => {
      const el = document.elementFromPoint(x, y);
      const button = el?.closest('button');
      return button?.textContent?.trim() || null;
    }, { x: centerX, y: centerY, expected: label });
    expect(hitLabel).toBe(label);
  }
}

test.describe('Deep Research review panel footer visibility', () => {
  const fixture = path.join(__dirname, 'fixtures', 'deep-research-review-panel.html');

  test('1280x800 keeps both action buttons in the viewport and clickable', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto(`file://${fixture}`);
    await assertFooterButtonsInViewportAndClickable(page);
    await page.screenshot({
      path: process.env.ASKPDF_E2E_SCREENSHOT_PATH || '/opt/cursor/artifacts/review-panel-1280x800-after.png',
      fullPage: false,
    });
  });

  test('1280x650 keeps both action buttons in the viewport and clickable', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 650 });
    await page.goto(`file://${fixture}`);
    await assertFooterButtonsInViewportAndClickable(page);
    await page.screenshot({
      path: '/opt/cursor/artifacts/review-panel-1280x650-after.png',
      fullPage: false,
    });
  });

  test('1024x768 keeps both action buttons in the viewport', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 });
    await page.goto(`file://${fixture}`);
    await assertFooterButtonsInViewportAndClickable(page);
  });
});
