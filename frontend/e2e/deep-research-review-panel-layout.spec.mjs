import { test, expect } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const LIMITATIONS_MIN_VISIBLE_PX = 48;

async function readVisibleBounds(page) {
  const secondary = await page.locator('.secondary-pane').boundingBox();
  const viewport = page.viewportSize();
  const pane = secondary ?? { x: 0, y: 0, width: viewport?.width || 0, height: viewport?.height || 0 };
  return { pane, viewport };
}

async function assertFooterButtonsInViewportAndClickable(page) {
  const finish = page.getByRole('button', { name: /Finish with this result/ });
  const continueBtn = page.getByRole('button', { name: 'Continue with guidance' });
  await expect(finish).toBeVisible();
  await expect(continueBtn).toBeVisible();

  const finishBox = await finish.boundingBox();
  const continueBox = await continueBtn.boundingBox();
  const { pane } = await readVisibleBounds(page);
  expect(finishBox).not.toBeNull();
  expect(continueBox).not.toBeNull();

  const paneBottom = pane.y + pane.height;
  expect(finishBox.y).toBeGreaterThanOrEqual(pane.y);
  expect(continueBox.y).toBeGreaterThanOrEqual(pane.y);
  expect(finishBox.y + finishBox.height).toBeLessThanOrEqual(paneBottom);
  expect(continueBox.y + continueBox.height).toBeLessThanOrEqual(paneBottom);

  for (const [label, box] of [['Finish with this result', finishBox], ['Continue with guidance', continueBox]]) {
    const centerX = box.x + box.width / 2;
    const centerY = box.y + box.height / 2;
    const hitLabel = await page.evaluate(({ x, y }) => {
      const el = document.elementFromPoint(x, y);
      const button = el?.closest('button');
      return button?.textContent?.trim() || null;
    }, { x: centerX, y: centerY });
    expect(hitLabel).toContain(label.split(' ')[0] === 'Finish' ? 'Finish' : 'Continue');
  }
}

async function assertLimitationsListVisible(page) {
  const limitations = page.locator('#limitations');
  await expect(limitations).toBeVisible();
  const box = await limitations.boundingBox();
  expect(box).not.toBeNull();
  expect(box.height).toBeGreaterThanOrEqual(LIMITATIONS_MIN_VISIBLE_PX);
  const { pane } = await readVisibleBounds(page);
  const paneBottom = pane.y + pane.height;
  const firstItem = page.locator('#limitations li').first();
  await expect(firstItem).toBeVisible();
  const firstBox = await firstItem.boundingBox();
  expect(firstBox).not.toBeNull();
  expect(firstBox.y).toBeGreaterThanOrEqual(pane.y);
  expect(firstBox.y + firstBox.height).toBeLessThanOrEqual(paneBottom);
}

async function assertReviewLayout(page) {
  await assertFooterButtonsInViewportAndClickable(page);
  await assertLimitationsListVisible(page);
}

test.describe('Deep Research review panel footer visibility', () => {
  const fixture = path.join(__dirname, 'fixtures', 'deep-research-review-panel.html');

  test('1280x800 keeps actions clickable and limitations visible', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto(`file://${fixture}`);
    await assertReviewLayout(page);
    await page.screenshot({
      path: process.env.ASKPDF_E2E_SCREENSHOT_PATH || '/opt/cursor/artifacts/review-panel-1280x800-after.png',
      fullPage: false,
    });
  });

  test('1280x650 keeps actions clickable and limitations visible', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 650 });
    await page.goto(`file://${fixture}`);
    await assertReviewLayout(page);
    await page.screenshot({
      path: '/opt/cursor/artifacts/review-panel-1280x650-after.png',
      fullPage: false,
    });
  });

  test('1024x768 stacked workbench keeps actions clickable and limitations visible', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 });
    await page.goto(`file://${fixture}`);
    await expect(page.locator('.workbench.stacked')).toHaveCount(1);
    await assertReviewLayout(page);
  });
});
