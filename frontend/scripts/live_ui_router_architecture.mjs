#!/usr/bin/env node
/** Minimal UI check: router workflow answers Transformer for architecture question. */
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

function indexingReady(status) {
  const parsing = String(status?.parsing?.status || "");
  const indexing = String(status?.indexing?.status || "");
  if (["completed", "ready"].includes(parsing) && ["completed", "ready"].includes(indexing)) return true;
  const projection = status?.document_processing?.projection || {};
  const vectors = String(projection.vector_status || "");
  if (projection.status === "completed" && ["ready", "indexed", "completed"].includes(vectors)) return true;
  return String(status?.status || "") === "ready";
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

async function api(method, path, body) {
  const res = await fetch(`${API}${path}`, {
    method,
    headers: { Authorization: `Bearer ${ADMIN}`, "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${method} ${path} ${res.status}`);
  return res.json();
}

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

try {
  const project = await api("POST", "/api/projects", { name: "ui-arch", description: "live", embedding_model: "BAAI/bge-m3" });
  const thread = await api("POST", `/api/projects/${project.id}/threads`, { name: "ui-arch" });
  await page.goto(APP, { waitUntil: "networkidle", timeout: 120000 });

  await uploadAndIndex(thread.id);
  await api("PUT", `/api/threads/${thread.id}/settings`, { agent_workflow: { workflow_id: "router_rag_agent" } });
  let chatBody = await api("POST", `/api/threads/${thread.id}/chat`, {
    thread_id: thread.id,
    question: "What architecture does the paper propose? Reply with the architecture name only.",
    llm_model: process.env.ASKPDF_E2E_LLM_MODEL || "google/gemini-2.5-flash",
    use_web_search: false,
    use_reranker: true,
    context_window: 64000,
  });
  let answer = String(chatBody.answer || chatBody.final_answer || "");
  const runId = String(chatBody.run_id || chatBody.agent_run_id || "");
  const deadline = Date.now() + 300_000;
  while (!answer.trim() && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 2000));
    if (runId) {
      const payload = await api("GET", `/api/agent-runs/${runId}?thread_id=${thread.id}`);
      const agentRun = payload.agent_run || payload;
      answer = String(agentRun.final_answer || answer);
      if (answer.trim()) break;
    }
    const msgs = await api("GET", `/api/threads/${thread.id}/messages?limit=8`);
    const list = Array.isArray(msgs) ? msgs : msgs.messages || [];
    for (const item of [...list].reverse()) {
      if (item?.role === "assistant" && String(item.content || "").trim()) {
        answer = String(item.content);
        break;
      }
    }
    if (answer.trim()) break;
  }
  await page.goto(APP, { waitUntil: "networkidle", timeout: 120000 });
  const threadRow = page.getByRole("button", { name: new RegExp(thread.name, "i") }).first();
  await threadRow.click({ timeout: 60_000 });
  await page.getByRole("tab", { name: /chat/i }).click({ timeout: 30_000 }).catch(() => {});
  const assistantReplyDeadline = Date.now() + 300_000;
  let visibleAnswer = "";
  while (Date.now() < assistantReplyDeadline) {
    const bodies = await page.locator(".MuiBox-root").allTextContents();
    visibleAnswer = bodies.join("\n").trim();
  if (/\btransformer\b/i.test(visibleAnswer) && !/technical error/i.test(visibleAnswer)) break;
    await page.waitForTimeout(4000);
    await page.reload({ waitUntil: "networkidle" });
    await threadRow.click({ timeout: 60_000 }).catch(() => {});
  }
  await page.waitForTimeout(2000);
  await page.screenshot({ path: `${ART}/live-router_rag_agent-architecture-ui.png`, fullPage: true });
  const merged = `${answer}\n${visibleAnswer}`;
  const ok = /\btransformer\b/i.test(merged) && !/does not specify/i.test(merged) && !/technical error/i.test(merged);
  console.log(JSON.stringify({ pass: ok, answer, thread_id: thread.id }, null, 2));
  process.exitCode = ok ? 0 : 1;
} finally {
  await browser.close();
}