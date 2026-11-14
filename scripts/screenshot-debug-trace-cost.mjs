import { chromium } from '@playwright/test';

const base = process.env.ASKPDF_E2E_BASE_URL || 'http://127.0.0.1:3000';
const out = process.env.ASKPDF_E2E_SCREENSHOT || '/opt/cursor/artifacts/debug-trace-cost-ui.png';
const threadId = process.env.ASKPDF_E2E_THREAD_ID || 'db2d0985-e0e3-4591-9560-0e77b7ea0e1a';

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
await page.goto(base, { waitUntil: 'domcontentloaded', timeout: 120000 });
await page.waitForTimeout(3000);

const threadLink = page.getByRole('button', { name: /hermes-e2e|leftovers|scope-live|verify/i }).first();
if (await threadLink.count()) {
  await threadLink.click();
  await page.waitForTimeout(2000);
}

await page.getByRole('tab', { name: 'Debug Trace' }).click({ timeout: 60000 });
await page.waitForTimeout(4000);
await expectCostVisible(page);
await page.screenshot({ path: out, fullPage: false });
await browser.close();

async function expectCostVisible(page) {
  const cost = page.getByText(/Provider cost|llm_provider_cost|USD|\$0\./i).first();
  await cost.waitFor({ timeout: 60000 });
}
