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
    const scroll = document.querySelector('[data-testid="review-decision-scroll"]');
    const footer = document.querySelector('[data-testid="review-decision-footer"]');
    if (!scroll || !footer) return null;
    return scroll.getBoundingClientRect().bottom - footer.getBoundingClientRect().top;
  });
  expect(overlapPx).not.toBeNull();
  expect(overlapPx).toBeLessThanOrEqual(0.5);
}

async function countFullyVisibleLimitations(page) {
  const pane = await readPaneBounds(page);
  const paneBottom = pane.y + pane.height;
  const footer = await page.getByTestId('review-decision-footer').boundingBox();
  const visibleCeiling = Math.min(paneBottom, footer?.y ?? paneBottom);
  const items = page.locator('#limitations li');
  let visibleCount = 0;
  const count = await items.count();
  for (let index = 0; index < count; index += 1) {
    const box = await items.nth(index).boundingBox();
    if (!box) continue;
    if (box.y >= pane.y && box.y + box.height <= visibleCeiling + 0.5) visibleCount += 1;
  }
  return visibleCount;
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
  await expect(items).toHaveCount(12);

  let visibleCount = 0;
  const count = await items.count();
  for (let index = 0; index < count; index += 1) {
    const box = await items.nth(index).boundingBox();
    if (!box) continue;
    const fullyVisible = box.y >= pane.y && box.y + box.height <= visibleCeiling + 0.5;
    if (fullyVisible) visibleCount += 1;
  }
  const viewportHeight = page.viewportSize()?.height ?? 800;
  const minVisible = viewportHeight <= 650 && count > 8 ? 1 : 2;
  expect(visibleCount).toBeGreaterThanOrEqual(minVisible);
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

async function assertAllLimitationsReachableByScroll(page) {
  const scroll = page.getByTestId('review-decision-scroll');
  const footer = await page.getByTestId('review-decision-footer').boundingBox();
  expect(footer).not.toBeNull();
  await scroll.evaluate((el) => { el.scrollTop = el.scrollHeight; });
  const last = page.locator('#limitations li').last();
  await expect(last).toBeVisible();
  const lastBox = await last.boundingBox();
  expect(lastBox).not.toBeNull();
  expect(lastBox.y + lastBox.height).toBeLessThanOrEqual(footer.y + 0.5);
}

async function assertTranscriptInsidePaneWhenExpanded(page) {
  const pane = await readPaneBounds(page);
  const paneBottom = pane.y + pane.height;
  const body = page.getByTestId('review-transcript-body');
  const bodyBox = await body.boundingBox();
  expect(bodyBox).not.toBeNull();
  expect(bodyBox.y).toBeGreaterThanOrEqual(pane.y);
  expect(bodyBox.y + bodyBox.height).toBeLessThanOrEqual(paneBottom + 0.5);
  const toggle = page.getByTestId('review-transcript-toggle');
  const toggleBox = await toggle.boundingBox();
  expect(toggleBox).not.toBeNull();
  expect(toggleBox.y + toggleBox.height).toBeLessThanOrEqual(paneBottom + 0.5);
  const footer = await page.getByTestId('review-decision-footer').boundingBox();
  expect(footer).not.toBeNull();
  expect(bodyBox.y + bodyBox.height).toBeLessThanOrEqual(footer.y + 0.5);
}

async function assertLimitationRowsDoNotPaintOverFooter(page) {
  await page.getByTestId('review-decision-scroll').evaluate((el) => { el.scrollTop = 0; });
  const footer = await page.getByTestId('review-decision-footer').boundingBox();
  const scroll = await page.getByTestId('review-decision-scroll').boundingBox();
  expect(footer).not.toBeNull();
  expect(scroll).not.toBeNull();

  const samples = await page.evaluate(({ footerTop, footerLeft, footerWidth, scrollBottom }) => {
    const hits = [];
    for (let x = footerLeft + 16; x < footerLeft + footerWidth - 16; x += 48) {
      for (const y of [footerTop + 8, footerTop + 28, footerTop + 52]) {
        const el = document.elementFromPoint(x, y);
        hits.push({
          x,
          y,
          limitation: Boolean(el?.closest('#limitations li')),
        });
      }
    }
    return hits;
  }, {
    footerTop: footer.y,
    footerLeft: footer.x,
    footerWidth: footer.width,
    scrollBottom: scroll.y + scroll.height,
  });

  for (const sample of samples) {
    expect(sample.limitation, `limitation row painted through footer at ${sample.x},${sample.y}`).toBe(false);
  }
}

async function assertTranscriptExpandedButtonsClickable(page) {
  const toggle = page.getByTestId('review-transcript-toggle-button');
  const body = page.getByTestId('review-transcript-body');
  await toggle.click();
  await expect(toggle).toHaveAttribute('aria-expanded', 'true');
  await expect(body).toBeVisible();
  await assertTranscriptInsidePaneWhenExpanded(page);
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
  await assertLimitationRowsDoNotPaintOverFooter(page);
  await assertAtLeastTwoLimitationItemsVisible(page);
  await assertAllLimitationsReachableByScroll(page);
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

  test('wide panel shows more limitations on first paint', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto(`file://${fixture}`);
    const narrowVisible = await countFullyVisibleLimitations(page);

    await page.setViewportSize({ width: 1600, height: 800 });
    const wideVisible = await countFullyVisibleLimitations(page);
    expect(wideVisible).toBeGreaterThan(narrowVisible);
    expect(wideVisible).toBeGreaterThanOrEqual(3);
  });
});
