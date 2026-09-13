import type { NextApiRequest, NextApiResponse } from "next";
import { Readable } from "node:stream";

const HOP_BY_HOP_HEADERS = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

export const config = {
  api: { bodyParser: false },
};

function headerEntries(headers: Headers): Array<[string, string]> {
  return Array.from(headers.entries()).filter(([name]) => !HOP_BY_HOP_HEADERS.has(name));
}

function backendUrl(req: NextApiRequest): URL {
  const backend = process.env.ASKPDF_BACKEND_URL?.trim();
  if (!backend) throw new Error("ASKPDF_BACKEND_URL is required for the frontend API proxy");

  const path = Array.isArray(req.query.path) ? req.query.path : [req.query.path];
  const target = new URL(`/${path.filter(Boolean).map((part) => encodeURIComponent(part)).join("/")}`, `${backend.replace(/\/+$/, "")}/`);
  for (const [key, value] of Object.entries(req.query)) {
    if (key === "path") continue;
    for (const item of Array.isArray(value) ? value : [value]) {
      if (item !== undefined) target.searchParams.append(key, item);
    }
  }
  return target;
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    const target = backendUrl(req);
    const headers = new Headers();
    for (const [name, value] of Object.entries(req.headers)) {
      if (name === "host" || name === "content-length" || HOP_BY_HOP_HEADERS.has(name)) continue;
      for (const item of Array.isArray(value) ? value : [value]) {
        if (item !== undefined) headers.append(name, item);
      }
    }

    const token = process.env.ASKPDF_ADMIN_TOKEN?.trim();
    if (!token) throw new Error("ASKPDF_ADMIN_TOKEN is required for the frontend API proxy");
    headers.set("authorization", `Bearer ${token}`);

    const method = req.method ?? "GET";
    const upstream = await fetch(target, {
      method,
      headers,
      body: method === "GET" || method === "HEAD" ? undefined : (req as unknown as BodyInit),
      // Node's fetch requires this when the request body is a streamed IncomingMessage.
      duplex: "half",
    } as RequestInit & { duplex: "half" });

    res.status(upstream.status);
    for (const [name, value] of headerEntries(upstream.headers)) res.setHeader(name, value);
    if (!upstream.body || method === "HEAD") {
      res.end();
      return;
    }
    Readable.fromWeb(upstream.body as never).pipe(res);
  } catch (error) {
    console.error("Frontend API proxy request failed", error);
    res.status(502).json({ detail: { code: "backend_unavailable", message: "Backend API unavailable" } });
  }
}
