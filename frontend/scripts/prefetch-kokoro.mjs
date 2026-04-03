import { KokoroTTS } from "kokoro-js";

/**
 * Build-time Kokoro prefetch script used by the frontend container image.
 * It initializes the configured model once so required artifacts are cached
 * before runtime requests hit the API.
 */
const modelId = process.env.KOKORO_MODEL_ID || "onnx-community/Kokoro-82M-v1.0-ONNX";
const dtype = process.env.KOKORO_DTYPE || "q8";
const device = process.env.KOKORO_DEVICE || "cpu";

/**
 * Initializes Kokoro and logs discovered voice count for build diagnostics.
 *
 * @returns Promise that resolves when model prefetch completes.
 */
async function main() {
  console.log(`Prefetching ${modelId} (dtype=${dtype}, device=${device})...`);
  const tts = await KokoroTTS.from_pretrained(modelId, { dtype, device });

  let voices = [];
  if (typeof tts.list_voices === "function") {
    voices = await tts.list_voices();
  } else if (typeof tts.listVoices === "function") {
    voices = await tts.listVoices();
  }

  console.log(`Kokoro prefetch complete. Voices: ${Array.isArray(voices) ? voices.length : 0}`);
}

main().catch((error) => {
  console.error("Kokoro prefetch failed:", error);
  process.exit(1);
});
