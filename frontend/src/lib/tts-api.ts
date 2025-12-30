const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "/api";

export async function getVoices(): Promise<string[]> {
    const res = await fetch(`${API_BASE}/voices`);
    if (!res.ok) {
        console.error("Failed to fetch voices");
        return [];
    }
    const data = await res.json();
    return data.voices;
}

export async function ttsSentence(text: string, voice: string, speed: number): Promise<{ audioUrl: string }> {
    const res = await fetch(`${API_BASE}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, voice, speed }),
    });

    if (!res.ok) {
        throw new Error("TTS failed");
    }

    return res.json();
}
