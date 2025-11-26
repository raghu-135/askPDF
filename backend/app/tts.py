import os
import tempfile
import soundfile as sf

from app.helper import load_text_to_speech, load_voice_style

# Initialize once at startup
# Initialize once at startup
ONNX_DIR = "/models/supertonic/onnx"   # mount or copy the ONNX files here
VOICE_STYLES_DIR = "/models/supertonic/voice_styles"

# Load TTS model and style
text_to_speech = load_text_to_speech(ONNX_DIR, use_gpu=False)

class SupertonicTTS:
    def __init__(self):
        self.tts = text_to_speech
        self.sample_rate = self.tts.sample_rate
        self.default_style = load_voice_style([os.path.join(VOICE_STYLES_DIR, "M1.json")], verbose=False)

    def synthesize(self, text: str, voice_style: str = "M1.json", speed: float = 1.0):
        # Construct full path to the voice style
        style_path = os.path.join(VOICE_STYLES_DIR, voice_style)
        
        if os.path.exists(style_path):
            style = load_voice_style([style_path], verbose=False)
        else:
            # Fallback to default if not found
            print(f"Voice style {voice_style} not found, using default.")
            style = self.default_style

        # Ensure speed is within reasonable bounds
        speed = max(0.5, min(speed, 2.0))

        wav, duration = self.tts(text, style, total_step=5, speed=speed)
        # Trim to duration
        audio = wav[0, : int(self.sample_rate * duration[0].item())]
        return audio, self.sample_rate

    def synthesize_to_file(self, text: str, out_path: str, voice_style: str = "M1.json", speed: float = 1.0) -> str:
        audio, sr = self.synthesize(text, voice_style, speed)
        sf.write(out_path, audio, sr, subtype="PCM_16")
        return out_path

_tts = SupertonicTTS()

def tts_sentence_to_wav(sentence_text: str, out_dir: str, voice_style: str = "M1.json", speed: float = 1.0) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=out_dir)
    os.close(fd)
    return _tts.synthesize_to_file(sentence_text, tmp, voice_style, speed)

def list_voice_styles():
    if not os.path.exists(VOICE_STYLES_DIR):
        return []
    files = [f for f in os.listdir(VOICE_STYLES_DIR) if f.endswith(".json")]
    return sorted(files)
