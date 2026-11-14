#!/usr/bin/env node
/** Minimal UI check: router workflow answers Transformer for architecture question. */
import { chromium } from "@playwright/test";
import { readFileSync } from "fs";
import { execFileSync } from "child_process";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(__dirname, "..");
const ART = process.env.ASKPDF_E2E_ARTIFACTS || "/opt/cursor/artifacts";

function loadAdmin() {
  const fromEnv = (process.env.ASKPDF_ADMIN_TOKEN || "").trim();
  if (fromEnv) return fromEnv;
  for (const line of readFileSync(resolve(REPO, ".env"), "utf8").split("\n")) {
    if (line.startsWith("ASKPDF_ADMIN_TOKEN=")) return line.slice("ASKPDF_ADMIN_TOKEN=".length).trim();
  }
  throw new Error("ASKPDF_ADMIN_TOKEN missing");
}

const ADMIN = loadAdmin();
const API = process.env.ASKPDF_E2E_API_URL || "http://127.0.0.1:8000";
const APP = process.env.ASKPDF_E2E_APP_URL || "http://127.0.0.1:3000";

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
  await page.goto(`${APP}/threads/${thread.id}`, { waitUntil: "domcontentloaded", timeout: 120000 });
  await page.waitForTimeout(3000);
  await page.screenshot({ path: `${ART}/live-router-ui-thread.png`, fullPage: true });

  await api("PUT", `/api/threads/${thread.id}/settings`, { agent_workflow: { workflow_id: "router_rag_agent" } });
  const chatBody = await api("POST", `/api/threads/${thread.id}/chat`, {
    thread_id: thread.id,
    question: "What architecture does the paper propose? Reply with the architecture name only.",
    llm_model: "openai/gpt-4.1-mini",
    use_web_search: false,
    use_reranker: true,
    context_window: 64000,
  });
  const answer = String(chatBody.answer || chatBody.final_answer || "");
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForTimeout(5000);
  await page.screenshot({ path: `${ART}/live-router_rag_agent-architecture-ui.png`, fullPage: true });
  const ok = /\btransformer\b/i.test(answer) && !/does not specify/i.test(answer);
  console.log(JSON.stringify({ pass: ok, answer, thread_id: thread.id }, null, 2));
  process.exitCode = ok ? 0 : 1;
} finally {
  await browser.close();
}