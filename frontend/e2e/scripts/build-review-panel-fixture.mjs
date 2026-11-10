/**
 * Builds e2e/fixtures/deep-research-review-panel.html using live capture geometry
 * from dom_snapshot_0a1b3790 (1280×800 / 1280×650 / 1024×768) without bundling that dump.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const REVIEW_DECISION_PANEL_LIMITATIONS_MIN_PX = 120;
const REVIEW_DECISION_PANEL_SCROLL_GRID_ROWS =
  `auto auto minmax(${REVIEW_DECISION_PANEL_LIMITATIONS_MIN_PX}px, 1fr) minmax(0, auto)`;
const REVIEW_DECISION_PANEL_SCROLL_MIN_PX = REVIEW_DECISION_PANEL_LIMITATIONS_MIN_PX + 56;
const REVIEW_DECISION_PANEL_FOOTER_MAX_HEIGHT = 'min(30vh, 136px)';
const REVIEW_PANEL_OPEN_TRANSCRIPT_MAX_PX = 72;
const REVIEW_STACKED_WORKBENCH_BOTTOM_RATIO = 0.58;
const WORKBENCH_AUTO_SIDE_MIN_WIDTH = 1100;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const outPath = path.join(__dirname, '..', 'fixtures', 'deep-research-review-panel.html');

/** Measured on live 1e858163 before height-budget fix (Documents tab, replayed review). */
const LIVE_LAYOUT = {
  sideRatio: 0.32,
  headerPx: 96,
  statusPx: 168,
  transcriptCapPx: REVIEW_PANEL_OPEN_TRANSCRIPT_MAX_PX,
};

const LIMITATION_LINES = [
  'Gap Analyze Model Architecture and Structural Weaknesses (todo-2)',
  'Gap Analyze Experimental Setup, Hyperparameters, and Evaluation Gaps (todo-3)',
  'Gap Synthesize and Refine Twelve Distinct Limitations (todo-4)',
  'Required document evidence was not established.',
  'The report makes detailed claims without manifest support.',
  'The report fails to address known result gaps todo-2, todo-3, todo-4.',
  'Grounding requirement for attached document was not satisfied.',
  'Evidence critic flagged missing source in manifest.',
];

const html = `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Deep Research review panel layout fixture</title>
    <style>
      * { box-sizing: border-box; }
      body { margin: 0; font-family: system-ui, sans-serif; font-size: 13px; background: #fafafa; }
      .app-shell { display: flex; height: 100vh; width: 100vw; overflow: hidden; }
      .primary-doc { background: #eee; min-height: 0; min-width: 0; }
      .secondary-pane { min-height: 0; min-width: 0; display: flex; flex-direction: column; overflow: hidden; }
      @media (min-width: ${WORKBENCH_AUTO_SIDE_MIN_WIDTH}px) {
        .app-shell { flex-direction: row; }
        .primary-doc { flex: 1 1 ${(1 - LIVE_LAYOUT.sideRatio).toFixed(2)}; }
        .secondary-pane { flex: 0 0 ${(LIVE_LAYOUT.sideRatio * 100).toFixed(0)}%; }
      }
      @media (max-width: ${WORKBENCH_AUTO_SIDE_MIN_WIDTH - 1}px) {
        .app-shell { flex-direction: column; }
        .primary-doc { flex: 1 1 ${(1 - REVIEW_STACKED_WORKBENCH_BOTTOM_RATIO).toFixed(2)}; }
        .secondary-pane { flex: 0 0 ${(REVIEW_STACKED_WORKBENCH_BOTTOM_RATIO * 100).toFixed(0)}%; }
        .header { height: 72px; }
        .status { height: 112px; }
      }
      .workbench { display: flex; flex-direction: column; height: 100%; padding: 8px; overflow: hidden; }
      .header { flex-shrink: 0; height: ${LIVE_LAYOUT.headerPx}px; border: 1px solid #ddd; background: #fff; padding: 8px; margin-bottom: 8px; }
      .status { flex-shrink: 0; height: ${LIVE_LAYOUT.statusPx}px; overflow: hidden; margin-bottom: 8px; }
      .alert { background: #e3f2fd; border: 1px solid #90caf9; padding: 8px; margin-bottom: 8px; }
      .objective { background: #1565c0; color: #fff; padding: 10px; border-radius: 6px; margin-top: 8px; }
      .transcript-shell {
        flex: 0 0 auto;
        flex-shrink: 0;
        max-height: ${LIVE_LAYOUT.transcriptCapPx}px;
        min-height: 48px;
        min-width: 0;
        overflow: hidden;
        display: flex;
        flex-direction: column;
      }
      .workbench.stacked .transcript-shell { order: 2; }
      .workbench.stacked .decision-wrap { order: 1; }
      .transcript { flex: 1 1 0; min-height: 0; overflow: auto; background: #fff; border: 1px solid #ddd; }
      .decision-wrap {
        flex: 1 1 0;
        min-height: 0;
        overflow: hidden;
        display: flex;
        flex-direction: column;
      }
      .decision-shell {
        display: flex;
        flex-direction: column;
        flex: 1 1 0;
        min-height: 0;
        overflow-y: auto;
        overflow-x: hidden;
        border: 1px solid #ddd;
        border-top: 2px solid #0288d1;
        background: #fff;
      }
      .decision-scroll {
        display: grid;
        grid-template-rows: ${REVIEW_DECISION_PANEL_SCROLL_GRID_ROWS};
        flex: 1 0 auto;
        min-height: ${REVIEW_DECISION_PANEL_SCROLL_MIN_PX}px;
      }
      .decision-title { text-align: center; padding: 8px; font-weight: 700; color: #666; }
      #review-limitations-scroller {
        min-height: ${REVIEW_DECISION_PANEL_LIMITATIONS_MIN_PX}px;
        height: 100%;
        overflow-y: auto;
        padding: 0 8px 8px;
      }
      .decision-body { min-height: 0; overflow: auto; padding: 8px; }
      #review-decision-footer {
        flex-shrink: 0;
        position: sticky;
        bottom: 0;
        z-index: 2;
        padding: 8px;
        border-top: 1px solid #ddd;
        background: #fafafa;
        max-height: ${REVIEW_DECISION_PANEL_FOOTER_MAX_HEIGHT};
        overflow: hidden;
        display: flex;
        flex-direction: column;
        box-shadow: 0 -4px 12px rgba(0,0,0,0.06);
      }
      #review-decision-footer .guidance-block {
        flex: 1 1 auto;
        min-height: 0;
        max-height: 72px;
        overflow-y: auto;
      }
      #review-decision-footer .actions { flex-shrink: 0; }
      .resize-handle { height: 6px; cursor: ns-resize; background: #ddd; }
      .workbench.stacked .resize-handle { display: none; }
      #guidance { width: 100%; padding: 6px 8px; border: 1px solid #ccc; border-radius: 4px; font: inherit; max-height: 88px; }
      #guidance.compact { height: 36px; }
      #guidance.expanded { height: 36px; }
      .actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
      .workbench.stacked .actions button { flex: 1 1 calc(50% - 6px); width: auto; }
      button { width: 100%; padding: 6px 12px; font: inherit; border-radius: 4px; }
      #finish-btn { background: #1976d2; color: #fff; border: none; }
      #continue-btn { background: #fff; border: 1px solid #1976d2; color: #1976d2; }
      .limitations li { margin-bottom: 6px; }
    </style>
  </head>
  <body>
    <div class="app-shell">
      <div class="primary-doc" aria-hidden="true">Documents viewer</div>
      <div class="secondary-pane">
        <div class="workbench" id="panel-root">
          <div class="header">CONTEXT 8,192 · LLM · Deep Research Agent</div>
          <div class="status">
            <div class="alert">Model, agent, and context window are locked for this task.</div>
            <div class="objective">objective · Awaiting your decision — critique limitations…</div>
          </div>
          <div class="decision-wrap">
            <div class="decision-shell" id="decision-panel">
              <div class="decision-scroll" id="review-decision-scroll" data-testid="review-decision-scroll">
                <div class="resize-handle" id="resize-handle" role="separator" aria-label="Resize decision panel"></div>
                <div class="decision-title">Research budget reached</div>
                <div id="review-limitations-scroller" data-testid="review-limitations-scroller">
                  <ul class="limitations" id="limitations">
                    ${LIMITATION_LINES.map((line) => `<li>${line}</li>`).join('\n                    ')}
                  </ul>
                </div>
                <div class="decision-body"><strong>Provisional answer</strong></div>
              </div>
              <div id="review-decision-footer" data-testid="review-decision-footer">
                <div class="guidance-block">
                  <label for="guidance">Guidance for the next research tranche</label>
                  <input id="guidance" class="compact" type="text" aria-label="Guidance for the next research tranche" />
                </div>
                <div class="actions">
                  <button type="button" id="finish-btn">Finish with this result (8 limitations)</button>
                  <button type="button" id="continue-btn">Continue with guidance (another tranche)</button>
                </div>
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
        const decisionWrap = document.querySelector('.decision-wrap');
        const guidance = document.getElementById('guidance');
        let panelRatio = 0.58;
        const decisionScroll = document.getElementById('review-decision-scroll');
        const syncStacked = () => {
          const stacked = window.innerWidth < ${WORKBENCH_AUTO_SIDE_MIN_WIDTH};
          workbench.classList.toggle('stacked', stacked);
          transcriptShell.style.flex = '0 0 auto';
          transcriptShell.style.maxHeight = '${LIVE_LAYOUT.transcriptCapPx}px';
          decisionWrap.style.flex = '1 1 0';
        };
        syncStacked();
        window.addEventListener('resize', syncStacked);
        guidance.addEventListener('focus', () => {
          guidance.classList.remove('compact');
          guidance.classList.add('expanded');
          guidance.style.height = '36px';
        });
        guidance.addEventListener('blur', () => {
          if (!guidance.value.trim()) {
            guidance.classList.remove('expanded');
            guidance.classList.add('compact');
          }
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
          decisionScroll.style.minHeight = Math.round(120 + panelRatio * 320) + 'px';
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

fs.writeFileSync(outPath, html);
console.log('wrote', outPath);
