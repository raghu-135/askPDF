/**
 * Builds e2e/fixtures/chat-web-search-approval-panel.html for layout regression.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REVIEW_DECISION_PANEL_FOOTER_MAX_HEIGHT = 'min(30vh, 136px)';
const REVIEW_TRANSCRIPT_EXPANDED_MAX_HEIGHT = 'min(28vh, 220px)';
const REVIEW_DECISION_COLUMN_MIN_HEIGHT = 'min(36vh, 240px)';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const outPath = path.join(__dirname, '..', 'fixtures', 'chat-web-search-approval-panel.html');

const html = `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Chat web search approval panel fixture</title>
    <style>
      * { box-sizing: border-box; }
      body { margin: 0; font-family: system-ui, sans-serif; font-size: 13px; background: #fafafa; height: 100vh; overflow: hidden; }
      .panel { display: flex; flex-direction: column; height: 100vh; padding: 8px; overflow: hidden; }
      .header { flex-shrink: 0; height: 72px; border: 1px solid #ddd; background: #fff; margin-bottom: 8px; padding: 8px; }
      .decision-wrap { flex: 1 1 0; min-height: ${REVIEW_DECISION_COLUMN_MIN_HEIGHT}; display: flex; flex-direction: column; overflow: hidden; }
      .decision {
        flex: 1 1 0; min-height: 0; display: flex; flex-direction: column; border: 1px solid #2196f3; border-radius: 4px;
        background: #fafafa; margin: 0 8px 8px; overflow: hidden;
      }
      .decision-title { text-align: center; color: #666; font-weight: 700; padding: 8px; flex-shrink: 0; }
      .decision-scroll { flex: 1 1 0; min-height: 0; overflow-y: auto; padding: 8px; display: flex; flex-direction: column; gap: 12px; }
      .decision-footer {
        flex-shrink: 0; border-top: 1px solid #ddd; background: #fff; padding: 8px; max-height: ${REVIEW_DECISION_PANEL_FOOTER_MAX_HEIGHT};
        display: flex; flex-direction: column; gap: 8px; box-shadow: 0 -4px 12px rgba(0,0,0,0.06);
      }
      .transcript { flex: 0 0 auto; max-height: ${REVIEW_TRANSCRIPT_EXPANDED_MAX_HEIGHT}; min-height: 0; overflow: auto; border: 1px solid #eee; margin: 0 8px 8px; padding: 8px; background: #fff; }
      .composer { flex-shrink: 0; height: 88px; border: 1px solid #ddd; background: #fff; margin: 0 8px; padding: 8px; }
      button { font: inherit; padding: 6px 12px; border-radius: 4px; border: 1px solid #ccc; background: #1976d2; color: #fff; cursor: pointer; }
      button.outline { background: #fff; color: #333; }
      pre { margin: 0; padding: 8px; background: #f0f0f0; font-size: 11px; white-space: pre-wrap; }
    </style>
  </head>
  <body>
    <div class="panel">
      <div class="header">Chat header (model + web search mode)</div>
      <div class="decision-wrap">
        <div class="decision" data-testid="chat-approval-shell">
          <div class="decision-title">Approve tool use?</div>
          <div data-testid="chat-human-review-panel" class="decision-scroll">
            <p>search_web would query the live web for current information.</p>
            <pre>{
  "query": "What is the current USD to EUR exchange rate?"
}</pre>
          </div>
          <div data-testid="review-decision-footer" class="decision-footer">
            <div class="actions">
              <button type="button">Approve once</button>
              <button type="button" class="outline">Skip this tool for this run</button>
            </div>
          </div>
        </div>
      </div>
      <div class="transcript" data-testid="chat-transcript-capped">
        <div>User: What is the current USD to EUR exchange rate?</div>
        <div>Assistant: Waiting for your approval…</div>
      </div>
      <div class="composer">Composer</div>
    </div>
  </body>
</html>`;

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, html, 'utf8');
console.log('Wrote', outPath);