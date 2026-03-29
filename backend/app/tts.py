"""
TTS module: Proxies synthesis requests to the RAG service.
Utilizes a shared volume for efficient audio file access.
"""

import os
import httpx
import logging

logger = logging.getLogger(__name__)

RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL")

async def tts_sentence_to_wav(
    sentence_text: str,
    out_dir: str,
    voice_style: str = None,
    speed: float = 1.0
) -> str:
    """
    Call the RAG service to synthesize a sentence.
    The RAG service saves the result to the shared volume (/data/audio).
    """
    if not RAG_SERVICE_URL:
        raise RuntimeError("RAG_SERVICE_URL is not set.")

    payload = {
        "text": sentence_text,
        "voice": voice_style or "af_heart", # default fallback
        "speed": speed
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{RAG_SERVICE_URL}/synthesize-tts",
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            filename = data.get("filename")
            return os.path.join(out_dir, filename)
        except Exception as e:
            logger.error(f"TTS proxy synthesis failed: {e}")
            raise RuntimeError(f"TTS service failed: {e}")


async def list_voice_styles() -> list[str]:
    """Fetch available voice styles from the RAG service."""
    if not RAG_SERVICE_URL:
        return []
        
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{RAG_SERVICE_URL}/voices", timeout=10.0)
            response.raise_for_status()
            data = response.json()
            return data.get("voices", [])
        except Exception as e:
            logger.error(f"Failed to fetch voices from RAG service: {e}")
            return []
