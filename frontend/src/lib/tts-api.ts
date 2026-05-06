/**
 * Fetches available TTS voices from the unified frontend TTS API.
 * Accepts multiple payload shapes and normalizes the result to voice ids.
 *
 * @returns A normalized list of available voice ids.
 */
export async function getVoices(): Promise<string[]> {
    const res = await fetch(`/api/tts?action=voices`);
    if (!res.ok) {
        console.error("Failed to fetch voices");
        return [];
    }
    const data = await res.json();
    const raw = data?.voices;
    if (Array.isArray(raw)) {
        return raw
            .map((v: any) => String(v?.voice ?? v?.id ?? v))
            .map((v: string) => v.replace(/\.json$/i, "").replace(/\.pt$/i, ""))
            .filter(Boolean);
    }
    if (raw && typeof raw === "object") {
        return Object.keys(raw)
            .map((v) => v.replace(/\.json$/i, "").replace(/\.pt$/i, ""))
            .filter(Boolean);
    }
    return [];
}

/**
 * Requests synthesis for a sentence and returns a URL that can be assigned
 * directly to an HTML audio element.
 *
 * @param text - Sentence text to synthesize.
 * @param voice - Voice id to use.
 * @param speed - Playback/synthesis speed multiplier.
 * @returns Audio URL payload from the TTS API.
 */
export async function ttsSentence(text: string, voice: string, speed: number): Promise<{ audioUrl: string }> {
    const res = await fetch(`/api/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, voice, speed }),
    });

    if (!res.ok) {
        throw new Error("TTS failed");
    }

    return res.json();
}
