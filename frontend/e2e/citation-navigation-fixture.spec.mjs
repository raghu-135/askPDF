import { test, expect } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

test('citation navigation fixture keeps highlight after nested tab re-activation', async ({ page }) => {
  const fixture = path.join(__dirname, 'fixtures', 'citation-navigation.html');
  await page.goto(`file://${fixture}`);

  await page.getByRole('button', { name: 'Open citation from canvas' }).click();
  await expect(page.locator('#current-id')).toHaveText('227');
  await page.getByRole('button', { name: 'Nested tab onChange (same doc)' }).click();
  await expect(page.locator('#current-id')).toHaveText('227');
  await expect(page.locator('[data-pdf-highlight-box]')).toHaveCount(1);

  const screenshotPath = process.env.ASKPDF_E2E_SCREENSHOT_PATH
    || '/opt/cursor/artifacts/canvas-citation-highlight-proof.png';
  await page.screenshot({ path: screenshotPath, fullPage: true });
});
