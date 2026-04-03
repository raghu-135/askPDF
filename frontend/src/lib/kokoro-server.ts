import fs from "fs/promises";
import path from "path";
import { randomUUID } from "crypto";
import { KokoroTTS } from "kokoro-js";

/**
 * Server-side Kokoro helpers used by Next.js API routes.
 * This module centralizes model loading, voice discovery, and audio file management.
 */
const MODEL_ID = process.env.KOKORO_MODEL_ID || "onnx-community/Kokoro-82M-v1.0-ONNX";
const MODEL_DTYPE = process.env.KOKORO_DTYPE || "q8";
const MODEL_DEVICE = process.env.KOKORO_DEVICE || "cpu";
const AUDIO_DIR = process.env.KOKORO_AUDIO_DIR || "/tmp/kokoro-audio";

let ttsPromise: Promise<any> | null = null;
let voicesCache: string[] | null = null;

/**
 * Normalizes voice identifiers from different formats (`.pt`, `.json`) to canonical ids.
 *
 * @param voice - Raw voice identifier from request or Kokoro metadata.
 * @returns Canonical voice id (e.g. `af_heart`) or empty string.
 */
function normalizeVoice(voice?: string): string {
  if (!voice) return "";
  return voice.replace(/\.pt$/i, "").replace(/\.json$/i, "");
}

/**
 * Lazily initializes and memoizes a single Kokoro runtime instance for the process.
 *
 * @returns The initialized Kokoro runtime.
 */
async function getTts() {
  if (!ttsPromise) {
    ttsPromise = KokoroTTS.from_pretrained(MODEL_ID, {
      dtype: MODEL_DTYPE as any,
      device: MODEL_DEVICE as any,
    }) as Promise<any>;
  }
  return ttsPromise;
}

/**
 * Discovers available voices from Kokoro runtime APIs and internal properties.
 * Different `kokoro-js` builds expose voice metadata in different shapes, so this
 * function normalizes all supported variants into a flat string list.
 *
 * @returns Sorted unique voice ids available to the current runtime.
 */
async function discoverVoices(): Promise<string[]> {
  const tts = await getTts();
  const anyTts = tts as any;

  let rawVoices: unknown = [];
  if (typeof anyTts.list_voices === "function") {
    rawVoices = await anyTts.list_voices();
  } else if (typeof anyTts.listVoices === "function") {
    rawVoices = await anyTts.listVoices();
  }

  const pickVoiceId = (item: any): string => {
    if (typeof item === "string") return normalizeVoice(item);
    if (!item || typeof item !== "object") return "";
    const candidate =
      item.voice ??
      item.id ??
      item.key ??
      item.voice_id ??
      item.voiceId ??
      "";
    return normalizeVoice(String(candidate));
  };

  const normalizeCollection = (value: unknown): string[] => {
    if (!value) return [];
    if (Array.isArray(value)) {
      return value.map((v) => pickVoiceId(v)).filter(Boolean);
    }
    if (value instanceof Map) {
      return [...value.keys()].map((k) => normalizeVoice(String(k))).filter(Boolean);
    }
    if (typeof value === "object") {
      const obj = value as Record<string, unknown>;
      if (Array.isArray(obj.voices)) {
        return obj.voices.map((v) => pickVoiceId(v)).filter(Boolean);
      }
      return Object.keys(obj).map((v) => normalizeVoice(v)).filter(Boolean);
    }
    return [];
  };

  let voices: string[] = normalizeCollection(rawVoices);
  if (voices.length === 0 && rawVoices && typeof rawVoices === "object") {
    const obj = rawVoices as Record<string, unknown>;
    voices = [
      ...voices,
      ...normalizeCollection(obj.data),
      ...normalizeCollection(obj.results),
      ...normalizeCollection(obj.voices),
    ];
  }

  // Some kokoro-js builds expose voices on instance properties rather than return values.
  voices = [
    ...voices,
    ...normalizeCollection(anyTts.voices),
    ...normalizeCollection(anyTts.available_voices),
    ...normalizeCollection(anyTts.voiceMap),
    ...normalizeCollection(anyTts.voice_map),
    ...normalizeCollection(anyTts.voiceManager?.voices),
    ...normalizeCollection(anyTts.voice_manager?.voices),
  ];

  return [...new Set(voices)].sort();
}

/**
 * Lists available voices with process-level caching to avoid repeated
 * runtime discovery/logging on every synthesis request.
 *
 * @param forceRefresh - When true, bypasses cache and re-discovers voices.
 * @returns Sorted unique voice ids available to the current runtime.
 */
export async function listVoices(forceRefresh = false): Promise<string[]> {
  if (!forceRefresh && voicesCache) {
    return voicesCache;
  }
  voicesCache = await discoverVoices();
  return voicesCache;
}

/**
 * Synthesizes text into a WAV file in the configured audio directory.
 *
 * @param text - Input text to synthesize.
 * @param voice - Requested voice id.
 * @param speed - Synthesis speed multiplier.
 * @returns Generated audio id and the resolved voice id used for synthesis.
 */
export async function synthesizeToFile(
  text: string,
  voice: string,
  speed: number,
): Promise<{ id: string; resolvedVoice: string }> {
  const tts = await getTts();
  const voices = await listVoices();
  const preferred = normalizeVoice(voice);
  const fallback = voices.includes("af_heart")
    ? "af_heart"
    : (voices[0] || preferred || "af_heart");
  const resolvedVoice = voices.includes(preferred) ? preferred : fallback;

  const id = randomUUID();
  const filePath = path.join(AUDIO_DIR, `${id}.wav`);
  await fs.mkdir(AUDIO_DIR, { recursive: true });

  const anyTts = tts as any;
  const generated = await anyTts.generate(text, {
    voice: resolvedVoice,
    speed,
  });

  if (!generated || typeof generated.save !== "function") {
    throw new Error("kokoro-js output does not expose save()");
  }

  await generated.save(filePath);
  return { id, resolvedVoice };
}

/**
 * Resolves an audio id to the absolute WAV path used by the TTS route.
 *
 * @param id - Audio identifier returned by `synthesizeToFile`.
 * @returns Absolute file path for the generated WAV file.
 */
export async function getAudioFilePath(id: string): Promise<string> {
  return path.join(AUDIO_DIR, `${id}.wav`);
}
