import { test, expect } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function assertFooterButtonsInViewport(page) {
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
}

test.describe('Deep Research review panel footer visibility', () => {
  test('1024x768 keeps both action buttons in the viewport', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 });
    const fixture = path.join(__dirname, 'fixtures', 'deep-research-review-panel.html');
    await page.goto(`file://${fixture}`);
    await assertFooterButtonsInViewport(page);
    await page.screenshot({
      path: process.env.ASKPDF_E2E_SCREENSHOT_PATH || '/opt/cursor/artifacts/r73-review-1024x768.png',
      fullPage: false,
    });
  });

  test('default desktop height keeps both action buttons in the viewport', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1113 });
    const fixture = path.join(__dirname, 'fixtures', 'deep-research-review-panel.html');
    await page.goto(`file://${fixture}`);
    await assertFooterButtonsInViewport(page);
    await page.screenshot({
      path: '/opt/cursor/artifacts/r73-review-panel.png',
      fullPage: false,
    });
  });
});
