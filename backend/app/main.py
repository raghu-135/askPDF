import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .pdf_parser import pdf_bytes_to_text, extract_text_with_coordinates
from .nlp import split_into_sentences
from .tts import tts_sentence_to_wav

API_PREFIX = "/api"
AUDIO_DIR = "/data/audio"

app = FastAPI(title="PDF TTS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(AUDIO_DIR, exist_ok=True)

@app.post(f"{API_PREFIX}/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    
    # TODO: Use unique IDs per upload instead of overwriting
    pdf_path = os.path.join("/static", "uploaded.pdf")
    os.makedirs("/static", exist_ok=True)
    
    content = await file.read()
    with open(pdf_path, "wb") as f:
        f.write(content)
        
    text, char_map = extract_text_with_coordinates(content)
    
    sentences = split_into_sentences(text)
    
    # Map sentences to bounding boxes by finding each sentence in the extracted text
    current_idx = 0
    enriched_sentences = []
    
    for s in sentences:
        s_text = s["text"]
        
        start = text.find(s_text, current_idx)
        if start != -1:
            end = start + len(s_text)
            bboxes = char_map[start:end]
            s["bboxes"] = bboxes
            current_idx = end
        else:
            s["bboxes"] = []
            
        enriched_sentences.append(s)

    return {"sentences": enriched_sentences, "pdfUrl": "/uploaded.pdf"}

@app.post(f"{API_PREFIX}/tts")
async def synthesize_sentence(payload: dict):
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in payload.")
    path = tts_sentence_to_wav(text, AUDIO_DIR)
    rel = os.path.relpath(path, "/")
    url = f"/{rel}"
    return {"audioUrl": url}

# Serve audio files
app.mount("/data", StaticFiles(directory="/data"), name="data")

# Mount static files last to avoid shadowing API routes
if os.path.isdir("/static"):
    app.mount("/", StaticFiles(directory="/static", html=True), name="static")
