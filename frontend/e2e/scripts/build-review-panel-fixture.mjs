/**
 * Builds e2e/fixtures/deep-research-review-panel.html from the captured PR#81 DOM snapshot
 * (metrics flex ratios + replayed limitation copy). Run after updating dom_snapshot/.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  REVIEW_DECISION_PANEL_LIMITATIONS_MIN_PX,
  REVIEW_DECISION_PANEL_SCROLL_GRID_ROWS,
  REVIEW_PANEL_OPEN_TRANSCRIPT_MAX_PX,
  WORKBENCH_AUTO_SIDE_MIN_WIDTH,
  REVIEW_STACKED_WORKBENCH_BOTTOM_RATIO,
} from '../../src/lib/deep-research-review-panel-layout.ts';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const snapshotRoot = path.join(root, 'fixtures', 'dom_snapshot', 'dom_snapshot_0a1b3790');
const metricsPath = path.join(snapshotRoot, 'metrics', '1280x800_guidance_unfocused.json');
const interruptPath = path.join(snapshotRoot, 'replayed_api', 'pending_interrupt.json');
const outPath = path.join(root, 'fixtures', 'deep-research-review-panel.html');

const metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
const interrupt = JSON.parse(fs.readFileSync(interruptPath, 'utf8'));
const warningIssues = (interrupt.warnings || []).flatMap((warning) => warning.details?.issues || []);
const limitations = [...(interrupt.gaps || []), ...warningIssues].slice(0, 8);
const fallbackItems = [
  'Incomplete review coverage on PDF attachments.',
  'Unverified numeric claims in section 3.',
  'Missing cross-reference for appendix A.',
  'Benchmark units not normalized.',
  'Figure alignment approximate on scanned page.',
  'Web search disabled for external baselines.',
  'Open todo checkpoints when run paused.',
  'Truncated reread of appendix B.',
];
const items = [...limitations];
for (const line of fallbackItems) {
  if (items.length >= 8) break;
  items.push(line);
}

const decisionFlex = metrics?.decisionFlex?.split?.(' ')?.[0] || '0.58';
const transcriptFlex = metrics?.transcriptFlex?.split?.(' ')?.[0] || '0.42';

const html = `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Deep Research review panel (from dom_snapshot_0a1b3790)</title>
    <style>
      * { box-sizing: border-box; }
      body { margin: 0; font-family: system-ui, sans-serif; font-size: 13px; background: #fafafa; }
      .app-shell { display: flex; height: 100vh; width: 100vw; overflow: hidden; }
      .primary-doc { background: #eee; min-height: 0; min-width: 0; }
      .secondary-pane { min-height: 0; min-width: 0; display: flex; flex-direction: column; overflow: hidden; }
      @media (min-width: ${WORKBENCH_AUTO_SIDE_MIN_WIDTH}px) {
        .app-shell { flex-direction: row; }
        .primary-doc { flex: 1 1 ${(1 - Number(transcriptFlex) - Number(decisionFlex) + Number(transcriptFlex)).toFixed(2)}; }
        .secondary-pane { flex: 0 0 32%; }
      }
      @media (max-width: ${WORKBENCH_AUTO_SIDE_MIN_WIDTH - 1}px) {
        .app-shell { flex-direction: column; }
        .primary-doc { flex: 1 1 ${(1 - REVIEW_STACKED_WORKBENCH_BOTTOM_RATIO).toFixed(2)}; }
        .secondary-pane { flex: 0 0 ${(REVIEW_STACKED_WORKBENCH_BOTTOM_RATIO * 100).toFixed(0)}%; }
      }
      .workbench {
        display: flex;
        flex-direction: column;
        height: 100%;
        padding: 8px;
        overflow: hidden;
      }
      .workbench.stacked .transcript-shell { order: 2; }
      .workbench.stacked .decision-shell-wrap { order: 1; }
      .header, .status { flex-shrink: 0; margin-bottom: 8px; }
      .header { height: 96px; border: 1px solid #ddd; background: #fff; padding: 8px; }
      .alert { background: #e3f2fd; border: 1px solid #90caf9; padding: 8px; margin-bottom: 8px; }
      .transcript-shell {
        flex: ${transcriptFlex} 1 0;
        min-height: 48px;
        min-width: 0;
        overflow: hidden;
        display: flex;
        flex-direction: column;
      }
      .workbench.stacked .transcript-shell {
        flex: 0 0 auto;
        max-height: ${REVIEW_PANEL_OPEN_TRANSCRIPT_MAX_PX}px;
      }
      .transcript { flex: 1 1 0; min-height: 0; overflow: auto; background: #fff; border: 1px solid #ddd; }
      .decision-shell-wrap {
        flex: ${decisionFlex} 1 0;
        min-height: 0;
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }
      .decision-shell {
        display: flex;
        flex-direction: column;
        flex: 1 1 0;
        min-height: 0;
        border: 1px solid #ddd;
        border-top: 2px solid #0288d1;
        background: #fff;
        overflow: hidden;
      }
      .decision-scroll {
        display: grid;
        grid-template-rows: ${REVIEW_DECISION_PANEL_SCROLL_GRID_ROWS};
        flex: 1 1 0;
        min-height: 0;
        overflow: auto;
      }
      .decision-title { text-align: center; padding: 8px; font-weight: 700; color: #666; }
      #review-limitations-scroller {
        min-height: ${REVIEW_DECISION_PANEL_LIMITATIONS_MIN_PX}px;
        max-height: min(36vh, 260px);
        overflow: auto;
        padding: 0 8px 8px;
      }
      .decision-body { min-height: 0; overflow: auto; padding: 8px; }
      #review-decision-footer {
        flex-shrink: 0;
        padding: 8px;
        border-top: 1px solid #ddd;
        background: #fafafa;
      }
      .resize-handle { height: 6px; cursor: ns-resize; background: #ddd; }
      #guidance { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; font: inherit; }
      #guidance.compact { padding: 4px 8px; height: 36px; }
      button { width: 100%; margin-top: 6px; padding: 6px 12px; font: inherit; border-radius: 4px; }
      #finish-btn { background: #1976d2; color: #fff; border: none; }
      #continue-btn { background: #fff; border: 1px solid #1976d2; color: #1976d2; }
      .workbench.stacked #review-decision-footer button { display: inline-block; width: calc(50% - 4px); }
      .limitations li { margin-bottom: 6px; }
    </style>
  </head>
  <body>
    <div class="app-shell" id="app-shell">
      <div class="primary-doc" aria-hidden="true">Documents viewer</div>
      <div class="secondary-pane">
        <div class="workbench" id="panel-root">
          <div class="header">CONTEXT 8,192 · LLM · Deep Research Agent</div>
          <div class="status">
            <div class="alert">Model, agent, and context window are locked for this task.</div>
          </div>
          <div class="decision-shell-wrap">
            <div class="decision-shell" id="decision-panel">
              <div class="decision-scroll">
                <div class="resize-handle" id="resize-handle" role="separator" aria-label="Resize decision panel"></div>
                <div class="decision-title">Research budget reached</div>
                <div id="review-limitations-scroller">
                  <ul class="limitations" id="limitations">
                    ${items.map((item) => {
                      const text = typeof item === 'string' ? item : (item.summary || item.body || item.title || String(item));
                      return `<li>${text.replace(/</g, '&lt;')}</li>`;
                    }).join('\n                    ')}
                  </ul>
                </div>
                <div class="decision-body"><strong>Provisional answer</strong></div>
              </div>
              <div id="review-decision-footer">
                <label for="guidance">Guidance for the next research tranche</label>
                <input id="guidance" class="compact" type="text" aria-label="Guidance for the next research tranche" />
                <button type="button" id="finish-btn">Finish with this result (8 limitations)</button>
                <button type="button" id="continue-btn">Continue with guidance (another tranche)</button>
              </div>
            </div>
          </div>
          <div class="transcript-shell" data-testid="review-transcript-frame">
            <div class="transcript" id="transcript"><div style="height:240px">Transcript</div></div>
          </div>
        </div>
      </div>
    </div>
    <script>
      (function () {
        const workbench = document.getElementById('panel-root');
        const transcriptShell = document.querySelector('.transcript-shell');
        const decisionWrap = document.querySelector('.decision-shell-wrap');
        const guidance = document.getElementById('guidance');
        let panelRatio = ${Number(decisionFlex) + Number(transcriptFlex) > 0 ? Number(decisionFlex) / (Number(decisionFlex) + Number(transcriptFlex)) : 0.58};
        const syncStacked = () => {
          const stacked = window.innerWidth < ${WORKBENCH_AUTO_SIDE_MIN_WIDTH};
          workbench.classList.toggle('stacked', stacked);
          if (stacked) {
            transcriptShell.style.flex = '0 0 auto';
            transcriptShell.style.maxHeight = '${REVIEW_PANEL_OPEN_TRANSCRIPT_MAX_PX}px';
            decisionWrap.style.flex = '1 1 0';
          } else {
            transcriptShell.style.maxHeight = '';
            transcriptShell.style.flex = (1 - panelRatio) + ' 1 0';
            decisionWrap.style.flex = panelRatio + ' 1 0';
          }
        };
        syncStacked();
        window.addEventListener('resize', syncStacked);

        guidance.addEventListener('focus', () => guidance.classList.remove('compact'));
        guidance.addEventListener('blur', () => {
          if (!guidance.value.trim()) guidance.classList.add('compact');
        });

        const handle = document.getElementById('resize-handle');
        let dragging = false;
        let startY = 0;
        let startRatio = panelRatio;
        handle.addEventListener('pointerdown', (event) => {
          if (workbench.classList.contains('stacked')) return;
          dragging = true;
          startY = event.clientY;
          startRatio = panelRatio;
          handle.setPointerCapture(event.pointerId);
        });
        handle.addEventListener('pointermove', (event) => {
          if (!dragging || workbench.classList.contains('stacked')) return;
          const panel = document.getElementById('panel-root');
          const delta = (startY - event.clientY) / panel.getBoundingClientRect().height;
          panelRatio = Math.max(0.16, Math.min(0.82, startRatio + delta));
          transcriptShell.style.flex = (1 - panelRatio) + ' 1 0';
          decisionWrap.style.flex = panelRatio + ' 1 0';
        });
        const stopDrag = (event) => {
          if (!dragging) return;
          dragging = false;
          if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
        };
        handle.addEventListener('pointerup', stopDrag);
        handle.addEventListener('pointercancel', stopDrag);
      })();
    </script>
  </body>
</html>
`;

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, html);
console.log('wrote', outPath);
