import { test, expect } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const fixture = path.join(__dirname, 'fixtures', 'deep-research-review-panel.html');
const buildScript = path.join(__dirname, 'scripts', 'build-review-panel-fixture.mjs');

const LONG_GUIDANCE =
  'Please tighten grounding against the attached paper. Cite every limitation with explicit page or section references.';

test.beforeAll(() => {
  execFileSync(process.execPath, [buildScript], { cwd: path.join(__dirname, '..') });
});

async function readPaneBounds(page) {
  const box = await page.locator('.secondary-pane').boundingBox();
  const viewport = page.viewportSize();
  return box ?? { x: 0, y: 0, width: viewport?.width || 0, height: viewport?.height || 0 };
}

async function assertNoFooterOverlapFirstPaint(page) {
  const overlapPx = await page.evaluate(() => {
    const scroller = document.querySelector('[data-testid="review-limitations-scroller"]');
    const footer = document.querySelector('[data-testid="review-decision-footer"]');
    if (!scroller || !footer) return null;
    return scroller.getBoundingClientRect().bottom - footer.getBoundingClientRect().top;
  });
  expect(overlapPx).not.toBeNull();
  expect(overlapPx).toBeLessThanOrEqual(0.5);
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
  const footer = await page.getByTestId('review-decision-footer').boundingBox();
  expect(footer).not.toBeNull();
  const visibleCeiling = Math.min(paneBottom, footer.y);

  await page.getByTestId('review-decision-scroll').evaluate((el) => { el.scrollTop = 0; });
  const items = page.locator('#limitations li');
  await expect(items).toHaveCount(8);

  let visibleCount = 0;
  const count = await items.count();
  for (let index = 0; index < Math.min(count, 8); index += 1) {
    const box = await items.nth(index).boundingBox();
    if (!box) continue;
    const fullyVisible = box.y >= pane.y && box.y + box.height <= visibleCeiling + 0.5;
    if (fullyVisible) visibleCount += 1;
  }
  expect(visibleCount).toBeGreaterThanOrEqual(2);
}

async function assertGuidanceStates(page) {
  const guidance = page.locator('#guidance');

  await guidance.blur();
  await assertNoFooterOverlapFirstPaint(page);
  await assertAtLeastTwoLimitationItemsVisible(page);
  await assertFooterButtonsInViewportAndClickable(page);

  await guidance.focus();
  await assertNoFooterOverlapFirstPaint(page);
  await assertAtLeastTwoLimitationItemsVisible(page);
  await assertFooterButtonsInViewportAndClickable(page);

  await guidance.fill(LONG_GUIDANCE);
  const continueBtn = page.getByRole('button', { name: /Continue with guidance/i });
  await expect(continueBtn).toBeEnabled();
  await assertNoFooterOverlapFirstPaint(page);
  await assertAtLeastTwoLimitationItemsVisible(page);
  await assertFooterButtonsInViewportAndClickable(page);

  await guidance.fill('');
  await guidance.blur();
  await expect(guidance).toHaveClass(/compact/);
}

async function assertTranscriptExpandedButtonsClickable(page) {
  const toggle = page.getByTestId('review-transcript-toggle-button');
  const body = page.getByTestId('review-transcript-body');
  await toggle.click();
  await expect(toggle).toHaveAttribute('aria-expanded', 'true');
  await expect(body).toBeVisible();
  await assertNoFooterOverlapFirstPaint(page);
  await assertFooterButtonsInViewportAndClickable(page);

  const guidance = page.locator('#guidance');
  await guidance.focus();
  await guidance.fill(LONG_GUIDANCE);
  await assertFooterButtonsInViewportAndClickable(page);
}

async function assertReviewLayout(page) {
  await expect(page.getByRole('separator', { name: 'Resize decision panel' })).toHaveCount(0);
  await assertNoFooterOverlapFirstPaint(page);
  await assertAtLeastTwoLimitationItemsVisible(page);
  await assertFooterButtonsInViewportAndClickable(page);
  await assertGuidanceStates(page);
  await assertTranscriptExpandedButtonsClickable(page);
}

const viewports = [
  { name: '1280x800', width: 1280, height: 800, stacked: false },
  { name: '1280x650', width: 1280, height: 650, stacked: false },
  { name: '1024x768 stacked', width: 1024, height: 768, stacked: true },
];

test.describe('Deep Research review panel layout fixture', () => {
  for (const viewport of viewports) {
    test(viewport.name, async ({ page }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.goto(`file://${fixture}`);
      if (viewport.stacked) {
        await expect(page.locator('.workbench.stacked')).toHaveCount(1);
      }
      await assertReviewLayout(page);
    });
  }
});
