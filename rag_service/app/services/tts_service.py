import os
import glob
import tempfile
import logging
import soundfile as sf
import torch
from kokoro import KPipeline

logger = logging.getLogger(__name__)

# Initialize the Kokoro pipeline once at module level
# Pointing to the local model weight to prevent runtime downloads
# This assumes the Dockerfile has downloaded the models to /models/kokoro
try:
    _pipeline = KPipeline(lang_code='a', model='/models/kokoro/kokoro-v0_19.pth')
except Exception as e:
    logger.error(f"Failed to load Kokoro pipeline: {e}")
    _pipeline = None

VOICES_DIR = "/models/kokoro/voices"

class KokoroTTS:
    """
    Core TTS engine using the Kokoro model for fast, high-quality speech synthesis.
    """
    def __init__(self):
        self.pipeline = _pipeline
        self.voices_dir = VOICES_DIR
        self.sample_rate = 24000  # Kokoro standard sample rate

    def get_available_voices(self) -> list[str]:
        """Fetch all .pt voice models available in the configured voices directory."""
        if not os.path.exists(self.voices_dir):
            return []
        pt_files = glob.glob(os.path.join(self.voices_dir, "*.pt"))
        return sorted([os.path.basename(f).replace(".pt", "") for f in pt_files])

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[torch.Tensor, int]:
        """
        Convert text to speech tensor using the specified voice and speed. 
        Falls back to the first available voice if the requested one is not found.
        """
        if not self.pipeline:
            raise RuntimeError("Kokoro pipeline not initialized.")
            
        voice_path = os.path.join(self.voices_dir, f"{voice}.pt")
        if not os.path.exists(voice_path):
            available = self.get_available_voices()
            if available:
                logger.warning(f"Voice {voice} not found, falling back to {available[0]}")
                voice_path = os.path.join(self.voices_dir, f"{available[0]}.pt")
            else:
                raise FileNotFoundError(f"No voices found in {self.voices_dir}")

        generator = self.pipeline(text, voice=voice_path, speed=speed)
        all_audio = [audio for _, _, audio in generator if audio is not None]
        if not all_audio:
            return torch.zeros(0), self.sample_rate
        full_audio = torch.cat(all_audio) if len(all_audio) > 1 else all_audio[0]
        return full_audio, self.sample_rate

    def synthesize_to_file(self, text: str, out_path: str, voice: str, speed: float = 1.0) -> str:
        """Synthesize speech and save it to a WAV file at the specified path."""
        audio, sr = self.synthesize(text, voice, speed)
        audio_np = audio.numpy() if torch.is_tensor(audio) else audio
        sf.write(out_path, audio_np, sr)
        return out_path

_tts = KokoroTTS()

def synthesize_speech(text: str, voice: str, speed: float = 1.0, out_dir: str = "/data/audio") -> str:
    """
    Synthesize speech and save the result as a WAV file in the shared storage volume.
    Returns the generated filename.
    """
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=out_dir)
    os.close(fd)
    
    available = _tts.get_available_voices()
    if not voice or voice.endswith(".json"):
        voice = available[0] if available else None
        
    if not voice:
        raise ValueError("No TTS voices available.")
        
    _tts.synthesize_to_file(text, tmp_path, voice, speed)
    return os.path.basename(tmp_path)

def list_voices() -> list[str]:
    """Expose available TTS voices for external API calls."""
    return _tts.get_available_voices()
