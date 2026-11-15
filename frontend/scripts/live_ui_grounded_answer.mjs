#!/usr/bin/env node
/** Capture a real UI screenshot with a grounded assistant answer (Attention PDF). */
import { chromium } from "@playwright/test";
import { readFileSync } from "fs";
import { execFileSync } from "child_process";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, "../..");
const ART = process.env.ASKPDF_E2E_ARTIFACTS || "/opt/cursor/artifacts";

function loadAdmin() {
  const fromEnv = (process.env.ASKPDF_ADMIN_TOKEN || "").trim();
  if (fromEnv) return fromEnv;
  const envPath = resolve(REPO_ROOT, ".env");
  for (const line of readFileSync(envPath, "utf8").split("\n")) {
    if (line.startsWith("ASKPDF_ADMIN_TOKEN=")) return line.slice("ASKPDF_ADMIN_TOKEN=".length).trim();
  }
  throw new Error("ASKPDF_ADMIN_TOKEN missing");
}

const ADMIN = loadAdmin();
const API = process.env.ASKPDF_E2E_API_URL || "http://127.0.0.1:8000";
const APP = process.env.ASKPDF_E2E_APP_URL || "http://127.0.0.1:3000";
const PDF = process.env.ASKPDF_E2E_PDF || "/tmp/attention.pdf";
const FREE_LLM =
  process.env.ASKPDF_E2E_FREE_LLM_MODEL ||
  process.env.ASKPDF_E2E_LLM_MODEL ||
  "inclusionai/ling-3.2-flash:free";
const RUN_TAG = String(Date.now());

function indexingReady(status) {
  const parsing = String(status?.parsing?.status || "");
  const indexing = String(status?.indexing?.status || "");
  if (["completed", "ready"].includes(parsing) && ["completed", "ready"].includes(indexing)) return true;
  const projection = status?.document_processing?.projection || {};
  const vectors = String(projection.vector_status || "");
  return projection.status === "completed" && ["ready", "indexed", "completed"].includes(vectors);
}

async function api(method, path, body) {
  const res = await fetch(`${API}${path}`, {
    method,
    headers: { Authorization: `Bearer ${ADMIN}`, "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${method} ${path} ${res.status}`);
  return res.json();
}

async function uploadAndIndex(threadId) {
  const out = execFileSync(
    "curl",
    ["-sf", "-X", "POST", `${API}/api/threads/${threadId}/files/upload`, "-H", `Authorization: Bearer ${ADMIN}`, "-F", `file=@${PDF}`],
    { encoding: "utf8" },
  );
  const fileHash = JSON.parse(out).file_hash;
  const deadline = Date.now() + 900_000;
  while (Date.now() < deadline) {
    const st = await api("GET", `/api/threads/${threadId}/files/${fileHash}/status`);
    if (indexingReady(st)) return fileHash;
    await new Promise((r) => setTimeout(r, 4000));
  }
  throw new Error("indexing timeout");
}

async function waitRuntimeReady() {
  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    try {
      const res = await fetch("http://127.0.0.1:8100/readyz");
      if (res.ok) return;
    } catch {
      /* retry */
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
  throw new Error("langgraph runtime not ready");
}

async function pollAssistantAnswer(threadId, runId) {
  const deadline = Date.now() + 300_000;
  let answer = "";
  while (Date.now() < deadline) {
    if (runId) {
      const payload = await api("GET", `/api/agent-runs/${runId}?thread_id=${threadId}`);
      const agentRun = payload.agent_run || payload;
      answer = String(agentRun.final_answer || answer).trim();
      const status = String(agentRun.status || "").toLowerCase();
      if (answer && !/technical error/i.test(answer)) return answer;
      if (status === "completed" && answer) return answer;
    }
    const msgs = await api("GET", `/api/threads/${threadId}/messages?limit=8`);
    const list = Array.isArray(msgs) ? msgs : msgs.messages || [];
    for (const item of [...list].reverse()) {
      if (item?.role === "assistant") {
        answer = String(item.content || "").trim();
        if (answer && !/technical error/i.test(answer)) return answer;
      }
    }
    await new Promise((r) => setTimeout(r, 3000));
  }
  return answer;
}

const threadName = `ui-grounded-${RUN_TAG}`;
const projectName = `ui-grounded-${RUN_TAG}`;
const question =
  "How many encoder layers does the base Transformer use? Reply with the number only.";

await waitRuntimeReady();

const project = await api("POST", "/api/projects", {
  name: projectName,
  description: "ui proof",
  embedding_model: "BAAI/bge-m3",
});
const thread = await api("POST", `/api/projects/${project.id}/threads`, { name: threadName });
await uploadAndIndex(thread.id);
await api("PUT", `/api/threads/${thread.id}/settings`, { agent_workflow: { workflow_id: "router_rag_agent" } });

const chatBody = await api("POST", `/api/threads/${thread.id}/chat`, {
  thread_id: thread.id,
  question,
  llm_model: FREE_LLM,
  use_web_search: false,
  use_reranker: true,
  context_window: 64000,
});
const runId = String(chatBody.run_id || chatBody.agent_run_id || "");
const answer = await pollAssistantAnswer(thread.id, runId);
if (!/\b6\b/.test(answer) || /technical error/i.test(answer)) {
  console.log(JSON.stringify({ pass: false, answer, thread_id: thread.id, stage: "api" }, null, 2));
  process.exit(1);
}

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });

try {
  await page.goto(APP, { waitUntil: "networkidle", timeout: 120000 });
  const homeTab = page.getByRole("tab", { name: /^home$/i });
  if (await homeTab.count()) await homeTab.click();

  const projectEntry = page.getByText(projectName, { exact: false }).first();
  await projectEntry.click({ timeout: 60_000 });

  const threadEntry = page
    .locator("li, [role='button']")
    .filter({ hasText: threadName })
    .filter({ hasText: /msg/i })
    .first();
  await threadEntry.click({ timeout: 60_000 });

  const visibleDeadline = Date.now() + 120_000;
  let visible = "";
  while (Date.now() < visibleDeadline) {
    visible = (await page.locator("body").innerText()) || "";
    if (visible.includes(question.slice(0, 40)) && /\b6\b/.test(visible) && !/technical error/i.test(visible)) {
      break;
    }
    await page.waitForTimeout(2000);
  }

  const shot = `${ART}/ui-grounded-answer-visible.png`;
  await page.screenshot({ path: shot, fullPage: true });
  const ok =
    /\b6\b/.test(visible) &&
    !/technical error/i.test(visible) &&
    visible.toLowerCase().includes("encoder");
  console.log(
    JSON.stringify({ pass: ok, answer, thread_id: thread.id, screenshot: shot, llm_model: FREE_LLM }, null, 2),
  );
  process.exitCode = ok ? 0 : 1;
} finally {
  await browser.close();
}