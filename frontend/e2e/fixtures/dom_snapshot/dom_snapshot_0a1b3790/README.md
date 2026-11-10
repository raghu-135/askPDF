# Deep Research review-panel DOM snapshot (PR #81 `0a1b3790`)

Captured so a cloud agent can build an e2e fixture from the **real Documents-tab workbench DOM and CSS**, not a hand-written approximation.

## SHA and method

- Frontend built from `origin/pr81` (`cursor/fix-review-panel-footer-8db6`) at **`0a1b3790e8bc0329206479b5a45381c4af9cfa69`**.
- **Option A failed.** Postgres had no task in `awaiting_approval` / `awaiting_result_review`. Task `e239e046-628e-4439-8596-09cdaf0e3d97` is `completed` after the earlier Finish click; `pending_interrupt_json` was cleared to `{}`.
- **Option B was used:** Playwright route interception against the real running frontend at `http://127.0.0.1:3000`. The bundle, CSS, and Documents-tab workbench DOM are live from this SHA. Only API JSON is replayed.
- Replayed payloads started from the live rag-proxy responses for task `e239e046-628e-4439-8596-09cdaf0e3d97`, thread `c405b612-8e59-4ef2-bda6-5129f4efe8f5` (“REVIEWPANEL attention”), run `a899c5cb-ec75-4e31-bf77-e527c5f587d5`, plus the original `interrupt.requested` event and `orchestration_result.task_result` from Postgres. Status was rewritten to `awaiting_approval` / `awaiting_result_review` / run `awaiting_human`, with the same 8 limitations (4 gaps + 4 review findings). Capabilities had `task.result_review.respond` re-enabled (the live completed run returns `disabled_reason: run_terminal`). EventSource was stubbed so `intervention.responded` could not flip the UI back to completed. Finish/Continue were never clicked. Todos, thread list, and the PDF in the Documents pane are live data.

## What was captured

For **1280×800**, **1280×650**, and **1024×768**, in two states:

- `(a) guidance unfocused` — first paint after opening the Documents tab, before focusing the field (`compactUntilFocus`: single-line, no helper text, `guidanceH=40`).
- `(b) guidance focused` — field focused and the sentence “Please tighten grounding against the attached paper.” typed so Continue enables (needed to reproduce the 1280×650 hit-test miss). The field does not collapse on blur (`guidanceH=46`).

Each state has:

| Path | Contents |
| --- | --- |
| `html/<slug>.html` | `document.documentElement.outerHTML` after styles settled. `<link rel="stylesheet">` inlined, `<script>` stripped so the file can be opened as a static document. |
| `layout_trees/<slug>.json` | Ancestry from the decision-panel grid up to `<body>`, plus limitations/footer chains, key nodes, computed layout styles, and bounding rects. **This is the most reliable fixture source.** |
| `hit_tests/<slug>.json` | `elementFromPoint` at each button center, with tag, ancestry, and rect. See especially `1280x650_guidance_focused`. |
| `screenshots/<slug>.png` | Live viewport screenshot. |
| `metrics/<slug>.json` | Same measurement script as the earlier 0a1b3790 retest. |
| `replayed_api/` | Slim mutated API JSON (run `debug` traces stripped). |
| `root_cause.json` | Structured finding + `human_summary`. |
| `comparison.json` | Per-state deltas vs the earlier live retest. |
| `verification/` | Playwright re-measure of the saved HTML at the same viewport. |

## How closely this matches the earlier live 0a1b3790 numbers

The replayed Documents-tab layout **matches the earlier live retest** on the numbers that mattered:

| State | Earlier live (RETEST_0a1b3790) | This snapshot | Match |
| --- | --- | --- | --- |
| 1280×800 unfocused Finish | **597.2–627.9**, in VP, hits=true (compact first paint) | **597.2–627.9**, hits=true, `guidanceH=40` | Exact |
| 1280×800 limitations | 0/8 visible, visibleH=0 | 0/8, visibleH=0 | Exact |
| 1280×800 slice / ratio | slice 371.5, bottomRatio 0.939 | 371.5, 0.939 | Exact |
| 1280×650 focused Finish | **576.3–607.1**, in VP | **576.3–607.1** | Exact box |
| 1280×650 focused Continue | **613.1–643.8**, in VP, enabled, **hits=false**, topEl `DIV:CONTEXT 8,192 Select LLM…` | **613.1–643.8**, enabled, **hits=false**, same topEl | Exact |
| 1280×650 unfocused Finish | earlier “initial” was already 576.3 (field already expanded) | **510.2–540.9**, compact `guidanceH=40` | Expected: this is the true compact first paint |
| 1024×768 focused buttons | **646.0–676.7**, side by side, hits=true when enabled | **646.0–676.7**, sideBySide=true, hits=true | Exact |
| 1024 stacked / boost | decision before transcript; bottom slice **445.44px = 58%** | decisionBeforeTranscript=true; `css-6pnmcd` height **445.438px** at y=322.56 | Exact |
| 1280×800 focused Finish | earlier “typed” was 620.4 after other interactions on a long-lived page | **597.2–627.9** (footer grew upward from 141.5px to 223.3px; buttons stayed put) | Same compact-page geometry; not a regression of the replay |

Title is “Research budget reached” (result-review with `trigger: budget`, model tokens exhausted). Button labels: “Finish with this result (8 limitations)” and “Continue with guidance (another tranche)”.

## Root-cause finding

### Why the limitations row shows 0 of 8 items

Not a missing list. The list is in the DOM (`css-vtx4dx`, 8 items, used height 156–180px) but it is **covered and overflow-clipped**.

The collapsing ancestor is the decision-panel flex child **`MuiBox-root css-1ortyul`**:

- `flex: 0.58 1 0px; min-height: 0; overflow: hidden`
- Used height is only the leftover slice under the 72px transcript cap and the column chrome: **253.9px at 1280×800**, **166.1px at 1280×650** (top ≈427.5, bottom ≈593.7 at 650).

Inside it, the CSS grid **`MuiBox-root css-cscn5l`** (`REVIEW_DECISION_PANEL_GRID_ROWS`) computes to:

- 1280×800 unfocused: `24px 27.9px 120px 0px 141.5px` (grid height 245.9px)
- 1280×650 focused: `24px 27.9px 120px 0px 223.3px` (grid height **158.1px**)
- 1024×768 unfocused: `24px 27.9px 120px 0px 104.8px` (grid height 162.9px)

So: handle, title, **limitations floor 120px**, **body row collapsed to 0**, footer 105–223px. The grid is `overflow-y: auto; max-height: 100%; min-height: 0`. The sticky footer (`css-1js3606`, `position: sticky; bottom: 0; z-index: 2`) is **141.5px compact / 223.3px expanded** — at 1280×650 the footer is taller than the whole grid, so it covers the 120px limitations row. The limitations scroller ends up at **y≈665**, below the footer and below the `overflow: hidden` clip. Fully visible item count is 0 at every size captured.

### What intercepts Continue at 1280×650 (guidance focused)

Continue is enabled and its bounding box is in the viewport (y=613.1–643.8), but `elementFromPoint(1060.91, 628.47)` does **not** hit the button.

The **clipping ancestor is again `css-1ortyul`** (`overflow: hidden`, bottom **593.67**). Continue’s center (y=**628.47**) is below that clip. The element at that point is the right-column paper **`MuiPaper-root css-otllmr`**, whose text starts with “CONTEXT 8,192 Select LLM… Model, agent, and context window are locked…”. That paper is not an ancestor of the button; hit-testing falls through the clipped footer onto the column behind it.

Full ancestry is in `hit_tests/1280x650_guidance_focused.json`.

## file:// re-measure

`verification/` records opening each HTML file in Playwright at the same viewport. A full-page workbench snapshot does **not** reliably re-create the live button Y positions as a static `file://` document: the shell’s height chain and JS-driven layout (`useMediaQuery`, 100% / flex children) do not all survive without the running Next app. Treat `layout_trees/`, `metrics/`, and `screenshots/` as the source of truth for geometry. The HTML is still useful for class names, DOM structure, emotion style tags, and the real grid/footer markup.

One early inlining of 1280×800 did re-measure Finish at 597.2–627.9 (delta 0). That result was not stable across viewports or later inlining attempts, so it is not claimed here as a general guarantee.

## What's left out of the zip

No API keys, env files, Authorization headers, or the 1.6MB run `debug` traces. Zip is well under 20 MB.

## How to open a snapshot

```js
page.set_viewport_size({ width: 1280, height: 800 })
page.goto('file:///path/to/html/1280x800_guidance_unfocused.html')
```

Scripts are stripped, so the page will not re-hydrate or call the API. Prefer the JSON layout trees when pixel geometry is required.
