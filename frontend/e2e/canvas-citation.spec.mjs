import { test, expect } from '@playwright/test';

const baseUrl = process.env.ASKPDF_E2E_BASE_URL;

test.describe('canvas document citation', () => {
  test.skip(!baseUrl, 'Set ASKPDF_E2E_BASE_URL to a running askPDF stack (docker compose dev).');

  test('opens documents tab, scrolls, and draws highlight overlays from canvas citation', async ({ page }) => {
    const threadId = process.env.ASKPDF_E2E_THREAD_ID;
    const canvasId = process.env.ASKPDF_E2E_CANVAS_ID;
    test.skip(!threadId || !canvasId, 'Set ASKPDF_E2E_THREAD_ID and ASKPDF_E2E_CANVAS_ID for this smoke test.');

    await page.goto(`${baseUrl}/?thread=${threadId}`);
    await page.getByRole('tab', { name: /canvas/i }).click();
    await page.getByRole('button', { name: /open/i }).first().click({ timeout: 60_000 });
    await page.getByRole('tab', { name: /documents/i }).waitFor({ state: 'visible' });

    const citation = page.locator('[data-canvas-citation="document"]').first();
    await citation.click();

    await expect(page.getByRole('tab', { name: /documents/i })).toHaveAttribute('aria-selected', 'true');
    const highlights = page.locator('[data-pdf-highlight-box]');
    await expect(highlights).not.toHaveCount(0, { timeout: 15_000 });

    await page.screenshot({
      path: process.env.ASKPDF_E2E_SCREENSHOT_PATH || '/tmp/canvas-citation-highlight.png',
      fullPage: false,
    });
  });
});
