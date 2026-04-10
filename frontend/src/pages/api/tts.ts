import type { NextApiRequest, NextApiResponse } from "next";
import fs from "fs/promises";
import { getAudioFilePath, listVoices, synthesizeToFile } from "../../lib/kokoro-server";

/**
 * Unified TTS API response variants.
 */
type TtsResponse =
  | { audioUrl: string; voice?: string }
  | { voices: string[] }
  | { error: string };

/**
 * Unified TTS API handler.
 *
 * Supported operations:
 * - `GET /api/tts?action=voices`: returns available voice ids.
 * - `GET /api/tts?action=audio&id=<id>`: returns generated WAV bytes.
 * - `POST /api/tts`: synthesizes text and returns a playable audio URL.
 *
 * @param req - Next.js API request.
 * @param res - Next.js API response.
 * @returns API response based on method and `action` query.
 */
export default async function handler(req: NextApiRequest, res: NextApiResponse<TtsResponse>) {
  const action = Array.isArray(req.query.action) ? req.query.action[0] : req.query.action;

  if (req.method === "GET" && action === "voices") {
    try {
      const voices = await listVoices();
      res.setHeader("Cache-Control", "no-store");
      return res.status(200).json({ voices });
    } catch (error) {
      console.error("Failed to list Kokoro voices", error);
      return res.status(500).json({ voices: [] });
    }
  }

  if (req.method === "GET" && action === "audio") {
    const id = Array.isArray(req.query.id) ? req.query.id[0] : req.query.id;
    if (!id) {
      return res.status(400).json({ error: "Missing audio id" });
    }

    try {
      const audioPath = await getAudioFilePath(id);
      const wav = await fs.readFile(audioPath);
      res.setHeader("Content-Type", "audio/wav");
      res.setHeader("Cache-Control", "no-store");
      return res.status(200).end(wav);
    } catch (error) {
      console.error("Failed to read synthesized audio", error);
      return res.status(404).json({ error: "Audio not found" });
    }
  }

  if (req.method !== "POST") {
    res.setHeader("Allow", "GET, POST");
    return res.status(405).json({ error: "Method not allowed" });
  }

  const text = typeof req.body?.text === "string" ? req.body.text.trim() : "";
  const voice = typeof req.body?.voice === "string" ? req.body.voice : "";
  const speed = Number(req.body?.speed ?? 1.0);

  if (!text) {
    return res.status(400).json({ error: "Missing 'text' in payload." });
  }

  try {
    const { id, resolvedVoice } = await synthesizeToFile(text, voice, speed);
    return res.status(200).json({
      audioUrl: `/api/tts?action=audio&id=${encodeURIComponent(id)}`,
      voice: resolvedVoice,
    });
  } catch (error) {
    console.error("TTS synthesis failed", error);
    return res.status(500).json({ error: "TTS failed" });
  }
}
