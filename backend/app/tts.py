"""
TTS Proxy Module: Routes synthesis requests to the processing service.
Handles coordination between the backend API and the specialized TTS synthesis service.
"""

import os
import logging
from .service_client import ProcessingService

logger = logging.getLogger(__name__)

async def tts_sentence_to_wav(
    service_client: ProcessingService,
    sentence_text: str,
    out_dir: str,
    voice_style: str = None,
    speed: float = 1.0
) -> str:
    """
    Request sentence synthesis from the processing service.
    Returns the path to the synthesized audio file in the shared storage volume.
    """
    try:
        filename = await service_client.synthesize_tts(
            text=sentence_text,
            voice=voice_style or "af_heart",
            speed=speed
        )
        return os.path.join(out_dir, filename)
    except Exception as e:
        logger.error(f"TTS proxy synthesis failed: {e}")
        raise RuntimeError(f"TTS service failed: {e}")
