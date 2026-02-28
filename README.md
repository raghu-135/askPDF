# askpdf

A full-stack PDF reading assistant with **Text-to-Speech (TTS)**, **RAG (Retrieval Augmented Generation)**, **multi-agent AI chat**, and **reasoning trace support**—all designed to run privately and locally on your own machine. Upload a PDF, have it read aloud with synchronized text highlighting, and chat with your document using a LangGraph-powered orchestrator with an optional Intent Agent. Everything works for free using open-source models like Docker Model Runner, Ollama, or LMStudio—no cloud/subscriptions required.

## 🌟 Features

### 📄 Reading & TTS
- **Unified Experience**: Seamlessly switch between reading the PDF and listening to chat responses
- **Multiple PDF Tabs**: Open and switch between multiple PDFs using a tabbed interface at the top of the viewer
- **Intelligent Text Processing**: Robust sentence segmentation with support for Markdown and non-punctuated text
- **High-Quality TTS**: Local speech synthesis using [Kokoro-82M](https://github.com/hexgrad/kokoro)
- **Visual Tracking**: Synchronized sentence highlighting in PDF and message highlighting in Chat
- **Interactive Navigation**: Double-click any sentence in the PDF or any message in the Chat to start playback
- **Centralized Controls**: Unified player in the footer manages all audio sources (Speed 0.5x - 2.0x)

### 🤖 Multi-Agent AI Architecture
- **Orchestrator Agent**: A LangGraph-powered agent that plans, selects tools, and synthesizes answers across multiple iterations
- **Intent Agent** *(optional, per-thread)*: A lightweight pre-processing agent that captures and rewrites the user's question before passing it to the orchestrator — improving query clarity and search precision
- **Tool-Calling Agents**: The orchestrator selects from a rich catalog of tools each turn, including document search, conversation memory recall, web search, document listing, and clarification requests
- **Configurable Iterations**: Control how many tool-call steps the agent is allowed to perform, tunable globally and per-thread
- **Force Final Answer**: When the maximum iteration budget is exhausted the agent is forced to synthesize a final answer from all gathered evidence instead of looping indefinitely

### 🧠 Reasoning / Thinking Trace Support
- **Multi-Provider Extraction**: Automatically extracts chain-of-thought reasoning from responses, supporting structured blocks (Anthropic Claude, OpenAI `o`-series, Responses API) and `<think>` tags (DeepSeek, QwQ, Qwen3-Thinking)
- **Stored in Database**: Reasoning traces are persisted in SQLite alongside the answer and can be re-displayed after page reload
- **Shown in UI**: Expandable reasoning panel in chat bubbles lets you inspect the AI's internal thinking step-by-step

### 💬 RAG-Powered Chat, Threads & Semantic Memory
- **Threaded Chat**: Organize conversations into threads with persistent SQLite storage for messages and file associations
- **Per-Thread Collections**: Each thread has its own isolated vector collection in Qdrant, locked to a specific embedding model
- **Dual-Search Retrieval**: AI searches both document chunks AND past Q&A pairs (semantic memory) simultaneously
- **Semantic Recollection**: The UI highlights which past chat messages were "recalled" and used by the AI to answer the current question
- **Internet Search (DuckDuckGo)**: Optionally augment answers with live web search results for up-to-date or external information; web sources are stored in SQLite and displayed after page reload
- **Context Management**: Intelligent token budgeting that scales proportionally to the configured context window, ensuring the most relevant PDF chunks, recent history, and semantic memories fit the LLM context window

### ⚙️ Per-Thread Prompt & Behaviour Settings
- **Thread Settings Dialog**: A per-thread configuration panel accessible from the chat header lets you tune AI behaviour without touching code
- **System Role**: Customise the AI's persona and expertise focus for each thread (up to 500 chars)
- **Tool Instructions**: Override the default prompt for each individual tool (document search, web search, memory recall, etc.) to guide how the agent uses that tool (up to 500 chars per tool)
- **Custom Instructions**: Append freeform additional instructions to every prompt in the thread (up to 2 000 chars)
- **Max Iterations**: Set the maximum number of tool-use rounds the orchestrator may take (range: 1–30)
- **Intent Agent Toggle**: Enable or disable the Intent Agent per thread; also configure how many rewrite iterations it is allowed
- **Prompt Preview**: Live preview of the fully composed system prompt before saving, so you know exactly what the LLM will see
- **Persistent Settings**: All thread settings are saved to SQLite and restored automatically when returning to a thread
### 🌐 Internet Search (DuckDuckGo)

You can enable **Internet Search** in the chat panel to let the AI answer questions using both your PDF and live web results (via DuckDuckGo). This is useful for:

- Getting up-to-date facts, news, or background not present in your PDF
- Clarifying ambiguous or missing information

**How it works:**
- When enabled, the app performs a DuckDuckGo search for your question and injects the top results into the LLM's context window, along with PDF content.
- The LLM then answers using both sources.
- Web search results (source URLs and snippets) are stored in SQLite and Qdrant, so they are still visible in the chat after a page reload.
- When a message is deleted, its associated web search results are also removed from SQLite and Qdrant.

**Privacy:**
- All queries are sent to DuckDuckGo only when Internet Search is enabled.
- No PDF content is sent to DuckDuckGo—only your question.

**Rate Limits:**
- DuckDuckGo and other free search APIs may rate limit requests if used too frequently.
- If rate limited, the app will notify you and fall back to PDF-only answers.

**Model Compatibility:**
- Any OpenAI-compatible LLM can use this feature. The search results are injected as plain text context, so no special model/tool-calling support is required.

### 🎨 Modern UI
- **Unified Navigation**: Double-click sentences or chat bubbles to start reading immediately
- **Dynamic Visual Feedback**: PDF sentence highlighting and Chat bubble illumination during playback
- **Resizable Chat Panel**: Drag to adjust the chat interface width (300-800px)
- **Auto-Scroll**: Both PDF and Chat automatically keep the active being-read content in view
- **Model Selection**: Centralized embedding model selection and dynamic LLM discovery

### 🖥️ Private & Local Design
All features of this app are designed to run entirely on your own machine or laptop, using only local resources by default. Document processing, AI chat, TTS, and chat/thread management all happen locally—no data is sent to external servers unless you explicitly enable Internet Search.

**Privacy Note:**
- When Internet Search is enabled, *only your question* (not your PDF content or chat history) is sent to DuckDuckGo for web search. All other processing, including PDF parsing, vector search, LLM inference, and chat/thread/message storage, remains local and private.
- If Internet Search is disabled, no data ever leaves your machine.

You can use free, open-source models with Docker Model Runner, Ollama, or LMStudio, so there are no required cloud costs or subscriptions.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Docker Compose                                 │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│    Frontend     │    Backend      │   RAG Service   │       Qdrant          │
│   (Next.js)     │    (FastAPI)    │    (FastAPI)    │   (Vector DB)         │
│   Port: 3000    │   Port: 8000    │   Port: 8001    │   Port: 6333          │
└─────────────────┴─────────────────┴─────────────────┴───────────────────────┘
                                          │
                                          ▼
                            ┌──────────────────────────────────────────────┐
                            │         DMR / Ollama / LMStudio / LLM        │
                            │            (OpenAI-compatible)               │
                            │             Port: 12434 (default)            │
                            └──────────────────────────────────────────────┘
```

### Services Overview

| Service | Port | Description |
|---------|------|-------------|
| **Frontend** | 3000 | Next.js React app with PDF viewer, chat UI, and thread management |
| **Backend** | 8000 | FastAPI server for PDF processing and TTS |
| **RAG Service** | 8001 | FastAPI server for document indexing, AI chat, thread/message/file management |
| **Qdrant** | 6333 | Vector database for semantic and memory search |
| **DMR/Ollama/LMStudio** | 12434 | Local LLM server (external, user-provided) |


## 📋 Prerequisites

- **Docker** and **Docker Compose**
- **A local LLM runtime** — pick any one of:
  - **Docker Model Runner (DMR)** — built into Docker Desktop, no extra install needed
  - **Ollama** — lightweight CLI runtime, great model library
  - **LMStudio** — GUI app, easy model browsing and loading

### Required Models (on your LLM server)

You need one **chat model** and one **embedding model** loaded in whichever runtime you choose:

| Runtime | Chat model example | Embedding model example |
|---------|-------------------|------------------------|
| DMR | `ai/qwen3:latest` | `ai/nomic-embed-text-v1.5:latest` |
| Ollama | `llama3.2` | `nomic-embed-text` |
| LMStudio | any GGUF chat model | any GGUF embedding model |

You select both models inside the app after it starts — no hardcoding required.

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/raghu13590/askpdf.git
cd askpdf
```



### 2. Choose a Local LLM Runtime and Set Up Your `.env`

The app needs an **OpenAI-compatible LLM server** running on your machine for both chat and embeddings. Pick whichever option suits you best, then create a `.env` file at the project root with the shown value.

---

#### Option A: Docker Model Runner (DMR) *(built into Docker Desktop — easiest if you already use Docker)*

1. Open **Docker Desktop** and make sure the **Model Runner** feature is enabled (Settings → Features in development → Enable Docker Model Runner).
2. Pull the required models from the Docker Desktop UI **or** via the CLI:
   ```bash
   docker model pull ai/qwen3:latest              # LLM
   docker model pull ai/nomic-embed-text-v1.5:latest  # Embeddings
   ```
3. Verify both models appear as **Running** in Docker Desktop → Model Runner.
4. Create your `.env` file:
   ```env
   LLM_API_URL=http://host.docker.internal:12434
   ```

---

#### Option B: Ollama *(great for running many open-source models)*

> Requires Ollama **v0.1.34+** for OpenAI-compatible API support.

1. [Download and install Ollama](https://ollama.com/download) for your OS.
2. Pull the required models:
   ```bash
   ollama pull llama3.2          # or any chat model you prefer
   ollama pull nomic-embed-text  # embedding model
   ```
3. Ollama runs on port `11434` by default. Create your `.env` file:
   ```env
   LLM_API_URL=http://host.docker.internal:11434
   ```

---

#### Option C: LMStudio *(best if you prefer a GUI for browsing and loading models)*

1. [Download and install LMStudio](https://lmstudio.ai/).
2. Open LMStudio, search for and download:
   - A chat model (e.g. `Llama 3.2`, `Qwen 2.5`, or any GGUF model)
   - An embedding model (e.g. `nomic-embed-text`)
3. Go to **Local Server** in LMStudio and click **Start Server**. The default port is `1234`.
4. Create your `.env` file:
   ```env
   LLM_API_URL=http://host.docker.internal:1234/v1
   ```

---

> **Note:** After creating or editing `.env`, you must restart the containers for the change to take effect.

### 3. Start the Application

```bash
docker-compose up --build
```

### 4. Access the Application

- **Main App**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **RAG API**: http://localhost:8001
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 📖 Usage

### Using Threads & PDFs

1. **Manage Threads**: Use the Sidebar to create new threads or select existing ones.
2. **Select Embedding Model**: When creating a new thread, choose the embedding model. This model is locked to the thread for consistency.
3. **Upload PDFs**: Within a thread, click "Upload PDF". You can upload multiple PDFs to the same thread.
4. **Switch Tabs**: Different PDFs in the same thread appear as tabs at the top of the viewer.
5. **PDF Processing**: Each uploaded PDF is parsed, sentences extracted, and indexed for RAG within that thread's collection.

### Reading & TTS

1. **Play Audio**: Click "Play" at the top to start text-to-speech.
2. **Navigate**: Use playback controls or double-click any sentence in the PDF or any chat bubble to jump audio to that point.
3. **Adjust Voice**: Select different voice styles and adjust playback speed (0.5x to 2.0x).
4. **Auto-Scroll**: The app automatically keeps the current sentence in view.

### Chatting & Semantic Memory

1. **Select LLM Model**: Choose an LLM from the chat panel dropdown.
2. **(Optional) Enable Internet Search**: Toggle the "Use Internet Search" switch above the chat input to allow the AI to use live web results.
3. **Ask Questions**: Type your question. The AI will search both the current PDFs and past conversations in the current thread.
4. **Semantic Identification**: If the AI uses past conversations to answer, the relevant messages will glow with a purple border in the chat history.
5. **Follow-up**: The system maintains context for follow-up questions within the thread.
6. **Read AI Answers**: Double-click any assistant chat bubble to have the response read aloud.
7. **View Reasoning**: If the model emits a reasoning/thinking trace (e.g. DeepSeek's `<think>` blocks or Claude's extended thinking), an expandable panel appears in the chat bubble.
8. **Clarification**: When the agent is unsure of the intent, it may present multiple-choice clarification options—click one to continue.

### Thread Settings (Prompt Customisation)

1. Open any thread and click the **⚙ Settings** icon in the chat header.
2. Adjust the fields:
   - **System Role** — changes the AI's persona for this thread.
   - **Tool Instructions** — override how the AI uses each tool (document search, web search, memory recall, etc.).
   - **Custom Instructions** — extra instructions appended to every prompt.
   - **Max Iterations** — maximum number of tool-use rounds before a forced final answer.
   - **Intent Agent** — toggle on/off; configure its iteration budget.
3. Click **Prompt Preview** to see the exact system prompt the LLM will receive.
4. Click **Save** — settings are persisted per-thread in SQLite.

## 🛠️ Technology Stack

### Backend Service
| Technology | Purpose |
|------------|---------|
| **FastAPI** | Web framework for REST APIs |
| **PyMuPDF (fitz)** | PDF parsing with character-level coordinates |
| **spaCy** | NLP for sentence segmentation |
| **Kokoro** | Neural TTS with 82M parameters |

### RAG Service
| Technology | Purpose |
|------------|---------|
| **FastAPI** | Web framework |
| **LangChain** | LLM/Embedding integration |
| **LangGraph** | Stateful multi-agent workflow (Orchestrator + Intent Agent) |
| **Qdrant Client** | Vector database operations |
| **aiosqlite** | Async SQLite for threads, messages, settings, and web sources |

### Frontend
| Technology | Purpose |
|------------|---------|
| **Next.js** | React framework |
| **Material-UI (MUI)** | UI components |
| **react-pdf** | PDF rendering |
| **react-markdown** | Chat message rendering |

## 📁 Project Structure

```
askpdf/
├── docker-compose.yml          # Multi-service orchestration
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py             # FastAPI app, upload & TTS endpoints
│       ├── pdf_parser.py       # PyMuPDF text extraction with coordinates
│       ├── nlp.py              # spaCy sentence segmentation
│       └── tts.py              # Kokoro TTS synthesis
├── rag_service/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                 # FastAPI app, index, chat, thread, file, message, settings, and prompt endpoints
│   ├── rag.py                  # Document chunking & indexing (thread-aware)
│   ├── agent.py                # LangGraph multi-agent workflow (Orchestrator + Intent Agent + tools)
│   ├── reasoning.py            # Multi-provider reasoning/thinking trace extraction
│   ├── models.py               # LLM/Embedding model clients, constants, and config helpers
│   ├── database.py             # SQLite thread/message/file/settings management
│   └── vectordb/
│       ├── base.py             # Abstract vector DB interface
│       └── qdrant.py           # Qdrant adapter implementation (threaded collections, web-source storage)
└── frontend/
  ├── Dockerfile
  ├── package.json
  └── src/
    ├── pages/
    │   └── index.tsx       # Main application page
    ├── components/
    │   ├── PdfUploader.tsx     # File upload with model selection
    │   ├── PdfViewer.tsx       # PDF rendering with overlays
    │   ├── PlayerControls.tsx  # Audio playback controls
    │   ├── ChatInterface.tsx   # RAG chat UI (thread-aware, settings dialog, reasoning panel)
    │   ├── ThreadSidebar.tsx   # Thread management UI
    │   └── TextViewer.tsx      # Alternative text display
    └── lib/
      ├── api.ts          # Backend & RAG API client (thread/message/file/settings/prompt)
      └── tts-api.ts      # TTS API client
```
The application expects an OpenAI-compatible API at the URL specified by `LLM_API_URL` in your `.env` file (default: `http://host.docker.internal:12434`).
## 📝 API Reference

### Backend Service (Port 8000)

#### `POST /api/upload`
Upload a PDF and extract sentences with bounding boxes.

**Request:** `multipart/form-data`
- `file`: PDF file

All environment variables, including `LLM_API_URL`, are now managed via a `.env` file at the project root. This file is loaded by both Docker Compose and the Python services.
- `embedding_model`: Model name for RAG indexing

**Response:**
```json
{
  "sentences": [
    {
      "id": 0,
| `LLM_API_URL` | RAG Service | `http://host.docker.internal:12434` | LLM server URL (set in `.env`; change to `...:11434` for default Ollama) |
      "bboxes": [
        {"page": 1, "x": 72, "y": 700, "width": 50, "height": 12, "page_height": 792, "page_width": 612}
      ]
    }
  ],
  "pdfUrl": "/abc123.pdf"
}
```

#### `GET /api/voices`
List available TTS voice styles.

**Response:**
```json
{
  "voices": ["M1.json", "F1.json", "M2.json"]
}
```

#### `POST /api/tts`
Synthesize speech for text.

**Request:**
```json
{
  "text": "Text to synthesize",
  "voice": "M1.json",
  "speed": 1.0
}
```

**Response:**
```json
{
  "audioUrl": "/data/audio/tmp_xyz.wav"
}
```

### RAG Service (Port 8001)

#### `POST /index` (Legacy)
Index document text into vector database (legacy, single collection).

#### `POST /threads` / `GET /threads` / `PUT /threads/{id}` / `DELETE /threads/{id}`
Create, list, update, and delete chat threads. Each thread has its own context, files, and messages.

#### `POST /threads/{thread_id}/files`
Add a file to a thread and trigger background indexing. Associates PDFs with threads for context-aware chat.

#### `POST /threads/{thread_id}/chat`
Chat with a thread using the multi-agent orchestrator (and optional Intent Agent).

**Request:**
```json
{
  "thread_id": "abc123",
  "question": "What is this document about?",
  "llm_model": "ai/qwen3:latest",
  "use_web_search": false,
  "max_iterations": 10,
  "context_window": 128000,
  "use_intent_agent": true,
  "intent_agent_max_iterations": 1,
  "system_role_override": "Expert researcher",
  "tool_instructions_override": {},
  "custom_instructions_override": ""
}
```

**Response:**
```json
{
  "answer": "This document discusses...",
  "reasoning": "First I searched for...",
  "reasoning_available": true,
  "reasoning_format": "tagged_text",
  "used_chat_ids": ["msg1", "msg2"],
  "pdf_sources": [ ... ],
  "web_sources": [ ... ]
}
```

#### `GET /threads/{thread_id}/settings`
Get persisted prompt/behaviour settings for a thread.

#### `PUT /threads/{thread_id}/settings`
Update persisted settings for a thread.

**Request body fields (all optional):**

| Field | Type | Description |
|-------|------|-------------|
| `max_iterations` | int (1–30) | Max tool-call rounds for orchestrator |
| `system_role` | string (≤500 chars) | AI persona override |
| `tool_instructions` | object | Per-tool prompt overrides |
| `custom_instructions` | string (≤2000 chars) | Additional instructions appended to every prompt |
| `use_intent_agent` | bool | Enable/disable the Intent Agent |
| `intent_agent_max_iterations` | int (1–10) | Iteration budget for Intent Agent |

#### `GET /prompt-tools`
Returns the tool catalog (id, display name, description, default prompt) and current default thread settings. Used by the settings dialog to populate tool instruction editors.

#### `POST /prompt-preview`
Returns the fully composed system prompt that will be sent to the LLM, given a set of settings. Used for live preview in the settings dialog.

**Request:**
```json
{
  "context_window": 128000,
  "system_role": "Expert researcher",
  "tool_instructions": {},
  "custom_instructions": "",
  "use_web_search": false,
  "intent_agent_ran": true
}
```

**Response:**
```json
{ "prompt": "You are an Expert researcher..." }
```

#### `GET /threads/{thread_id}/messages` / `DELETE /messages/{message_id}`
List and delete messages in a thread. Deleting a message also removes associated web-search results from Qdrant.

#### `GET /models`
Fetch available models from LLM server.

#### `GET /health`
Health check endpoint.

## 🔧 Configuration

### Environment Variables

| Variable | Service | Default | Description |
|----------|---------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | Frontend | `http://localhost:8000` | Backend API URL |
| `NEXT_PUBLIC_RAG_API_URL` | Frontend | `http://localhost:8001` | RAG API URL |
| `RAG_SERVICE_URL` | Backend | `http://rag-service:8000` | Internal RAG service URL |
| `QDRANT_HOST` | RAG Service | `qdrant` | Qdrant hostname |
| `QDRANT_PORT` | RAG Service | `6333` | Qdrant port |
| `LLM_API_URL` | RAG Service | `http://host.docker.internal:12434` | LLM server URL (Change to `...:11434` for default Ollama) |
| `DEFAULT_TOKEN_BUDGET` | RAG Service | `128000` | Default context-window size in tokens |
| `DEFAULT_MAX_ITERATIONS` | RAG Service | `10` | Default max orchestrator tool-call rounds |
| `MIN_MAX_ITERATIONS` | RAG Service | `1` | Minimum allowed value for max iterations |
| `MAX_MAX_ITERATIONS` | RAG Service | `30` | Maximum allowed value for max iterations |
| `INTENT_AGENT_MAX_ITERATIONS` | RAG Service | `1` | Default iteration budget for the Intent Agent |
| `MAX_CUSTOM_INSTRUCTIONS_CHARS` | RAG Service | `2000` | Max characters for custom instructions |
| `MAX_SYSTEM_ROLE_CHARS` | RAG Service | `500` | Max characters for system role override |
| `MAX_TOOL_INSTRUCTION_CHARS` | RAG Service | `500` | Max characters per tool instruction override |
| `MAX_ITERATIONS_SUFFICIENT_COVERAGE` | RAG Service | `2` | Iteration threshold for "sufficient coverage" early-exit hint |
| `MAX_ITERATIONS_PROBABLY_SUFFICIENT_COVERAGE` | RAG Service | `4` | Iteration threshold for "probably sufficient" hint |
| `WEB_SEARCH_ITERATION_BONUS` | RAG Service | `2` | Extra iterations granted when web search is enabled |

### Voice Styles

Voice styles (voices) are handled by the Kokoro engine. Available options are discovered dynamically from the system and populated in the UI dropdown.

### TTS Parameters

In `backend/app/tts.py`:
- `total_step`: Diffusion steps (default: 5) - higher = better quality, slower
- `speed`: Playback speed (0.5 - 2.0)

## 🔄 Data Flow

### PDF Upload Flow
```
User uploads PDF
  ↓
Backend: Save PDF → Extract text + coordinates (PyMuPDF)
  ↓
Backend: Split into sentences (spaCy)
  ↓
Backend: Map sentences to bounding boxes
  ↓
Backend: Trigger async RAG indexing (per-thread if using threads)
  ↓
RAG Service: Chunk text → Generate embeddings → Store in Qdrant (threaded collections)
  ↓
Frontend: Display PDF with clickable sentence overlays
```

### Threaded Chat & Semantic Memory Flow
```
User creates/selects thread
  ↓
User asks question in thread
  ↓
RAG Service: [Optional] Intent Agent rewrites / clarifies question
  ↓
RAG Service: Orchestrator Agent begins tool-call loop (up to max_iterations)
  ↓
  ├── search_documents          → Qdrant: top-K PDF chunks for thread
  ├── search_conversation_history → Qdrant: semantic memory recall
  ├── search_web                → DuckDuckGo (if enabled); stored in SQLite + Qdrant
  ├── search_pdf_by_document    → targeted per-document search
  ├── list_uploaded_documents   → enumerate PDFs in thread
  └── ask_for_clarification     → present choices to user
  ↓
RAG Service: Force final answer when budget exhausted
  ↓
RAG Service: Extract reasoning trace (structured blocks or <think> tags)
  ↓
RAG Service: Store answer + reasoning + web_sources in SQLite
  ↓
Frontend: Display markdown answer, expandable reasoning panel, web source cards
```

### TTS Playback Flow
```
User clicks Play or double-clicks sentence
  ↓
Frontend: Request /api/tts with sentence text
  ↓
Backend: Kokoro synthesizes audio → WAV file
  ↓
Frontend: Play audio, highlight current sentence
  ↓
On audio end: Auto-advance to next sentence
```

## 🐳 Docker Details

The application uses Docker Compose with four services:

1. **frontend**: Next.js dev server with hot reload
2. **backend**: FastAPI with TTS models mounted (Supertonic cloned from HuggingFace at build)
3. **rag-service**: FastAPI with LangChain/LangGraph
4. **qdrant**: Official Qdrant image with persistent storage

### Volumes
- \`qdrant_data\`: Persistent vector storage
- Source directories mounted for development hot-reload

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project uses the following third-party technologies:
- [Kokoro](https://github.com/hexgrad/kokoro) - Text-to-speech model
- [spaCy](https://spacy.io/) - Natural language processing
- [LangChain](https://langchain.com/) - LLM framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Stateful AI workflows
- [Qdrant](https://qdrant.tech/) - Vector database
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Next.js](https://nextjs.org/) - React framework

## 🙏 Acknowledgments

- **hexgrad** for the amazing Kokoro-82M model
- **spaCy** for robust NLP capabilities
- **LangChain** team for the excellent LLM framework
- **Qdrant** for the powerful vector database
- The open-source community for all the amazing tools

## 📧 Contact

For questions, issues, or suggestions, please open an issue on the [GitHub repository](https://github.com/raghu13590/askpdf).
