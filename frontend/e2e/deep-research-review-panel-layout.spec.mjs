import { test, expect } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const fixture = path.join(__dirname, 'fixtures', 'deep-research-review-panel.html');
const buildScript = path.join(__dirname, 'scripts', 'build-review-panel-fixture.mjs');

test.beforeAll(() => {
  execFileSync(process.execPath, [buildScript], { cwd: path.join(__dirname, '..') });
});

async function readPaneBounds(page) {
  const secondary = await page.locator('.secondary-pane').boundingBox();
  const viewport = page.viewportSize();
  return secondary ?? { x: 0, y: 0, width: viewport?.width || 0, height: viewport?.height || 0 };
}

async function assertFooterButtonsInViewportAndClickable(page) {
  const finish = page.getByRole('button', { name: /Finish with this result/i });
  const continueBtn = page.getByRole('button', { name: /Continue with guidance/i });
  await expect(finish).toBeVisible();
  await expect(continueBtn).toBeVisible();

  const finishBox = await finish.boundingBox();
  const continueBox = await continueBtn.boundingBox();
  const pane = await readPaneBounds(page);
  const paneBottom = pane.y + pane.height;

  for (const [label, box] of [['Finish', finishBox], ['Continue', continueBox]]) {
    expect(box).not.toBeNull();
    expect(box.y).toBeGreaterThanOrEqual(pane.y);
    expect(box.y + box.height).toBeLessThanOrEqual(paneBottom + 0.5);
    const centerX = box.x + box.width / 2;
    const centerY = box.y + box.height / 2;
    const hitLabel = await page.evaluate(({ x, y }) => {
      const el = document.elementFromPoint(x, y);
      const button = el?.closest('button');
      return button?.textContent?.trim() || null;
    }, { x: centerX, y: centerY });
    expect(hitLabel).toMatch(label === 'Finish' ? /Finish/i : /Continue/i);
  }
}

async function assertAtLeastTwoLimitationItemsVisible(page) {
  const pane = await readPaneBounds(page);
  const paneBottom = pane.y + pane.height;
  const items = page.locator('#limitations li');
  await expect(items).toHaveCount(8);
  let visibleCount = 0;
  const count = await items.count();
  for (let index = 0; index < Math.min(count, 8); index += 1) {
    const box = await items.nth(index).boundingBox();
    if (!box) continue;
    const fullyVisible = box.y >= pane.y && box.y + box.height <= paneBottom;
    if (fullyVisible) visibleCount += 1;
  }
  expect(visibleCount).toBeGreaterThanOrEqual(2);
}

async function assertGuidanceFocusedContinueClickable(page) {
  const guidance = page.locator('#guidance');
  await guidance.focus();
  await guidance.fill('Please tighten grounding against the attached paper.');
  const continueBtn = page.getByRole('button', { name: /Continue with guidance/i });
  await expect(continueBtn).toBeEnabled();
  await assertFooterButtonsInViewportAndClickable(page);
  await guidance.fill('');
  await guidance.blur();
  await expect(guidance).toHaveClass(/compact/);
}

async function assertResizeChangesTranscriptHeight(page) {
  const transcriptShell = page.getByTestId('review-transcript-frame');
  const before = await transcriptShell.evaluate((node) => node.getBoundingClientRect().height);
  const handle = page.locator('#resize-handle');
  const handleBox = await handle.boundingBox();
  expect(handleBox).not.toBeNull();
  await page.mouse.move(handleBox.x + handleBox.width / 2, handleBox.y + handleBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(handleBox.x + handleBox.width / 2, handleBox.y - 80, { steps: 8 });
  await page.mouse.up();
  const after = await transcriptShell.evaluate((node) => node.getBoundingClientRect().height);
  expect(Math.abs(after - before)).toBeGreaterThan(8);
}

test.describe('Deep Research review panel (dom_snapshot fixture)', () => {
  test('1280x800', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto(`file://${fixture}`);
    await assertAtLeastTwoLimitationItemsVisible(page);
    await assertFooterButtonsInViewportAndClickable(page);
    await assertGuidanceFocusedContinueClickable(page);
    await assertResizeChangesTranscriptHeight(page);
  });

  test('1280x650', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 650 });
    await page.goto(`file://${fixture}`);
    await assertAtLeastTwoLimitationItemsVisible(page);
    await assertFooterButtonsInViewportAndClickable(page);
    await assertGuidanceFocusedContinueClickable(page);
    await assertResizeChangesTranscriptHeight(page);
  });

  test('1024x768 stacked', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 });
    await page.goto(`file://${fixture}`);
    await expect(page.locator('.workbench.stacked')).toHaveCount(1);
    await assertAtLeastTwoLimitationItemsVisible(page);
    await assertFooterButtonsInViewportAndClickable(page);
    await assertGuidanceFocusedContinueClickable(page);
  });
});
