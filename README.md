# askPDF

A **private, local PDF research assistant** that reads your documents aloud and answers questions about them. Upload PDFs or capture webpages, then chat with your content using AI—all running on your own machine, no subscriptions required.

---

## 🚀 Quick Start

### 1. Install Prerequisites
- **Docker** and **Docker Compose**
- **A local LLM runtime** (choose one):
  - **Docker Model Runner** (built into Docker Desktop)
  - **Ollama** (lightweight CLI)
  - **LMStudio** (GUI app)

### 2. Set Up Your LLM

<details>
<summary>📖 Choose your LLM setup (click to expand)</summary>

- copy .env.example file and rename it to .env file.
- Pick an LLM server from below and set "LLM_API_URL' with the url accordingly

#### Option A: Docker Model Runner (Easiest)
1. Enable **Model Runner** in Docker Desktop → Settings → Features
2. Pull models:
   ```bash
   docker model pull ai/qwen3:latest
   docker model pull ai/nomic-embed-text-v1.5:latest
   ```
3. Set LLM_API_URL in `.env` file to:
   ```env
   LLM_API_URL=http://host.docker.internal:12434
   ```

#### Option B: Ollama
1. [Install Ollama](https://ollama.com/download)
2. Pull models:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
3. Set LLM_API_URL in `.env` file to:
   ```env
   LLM_API_URL=http://host.docker.internal:11434
   ```

#### Option C: LMStudio
1. [Install LMStudio](https://lmstudio.ai/)
2. Download a chat model and embedding model
3. Start Local Server (port 1234)
4. Set LLM_API_URL in `.env` file to:
   ```env
   LLM_API_URL=http://host.docker.internal:1234/v1
   ```

</details>

### 3. Start the App
```bash
docker-compose up --build
```

### 4. Use It!
- **Open**: http://localhost:3000
- **Upload PDFs** or add webpages
- **Click Play** to hear documents read aloud
- **Ask questions** about your content

---

## 🌟 What You Can Do

### � Read & Listen
- **Text-to-Speech**: High-quality voice reads your PDFs aloud
- **Sentence Tracking**: Visual highlighting shows what's being read
- **Multiple Documents**: Switch between PDFs and webpages with tabs
- **PDF Annotations**: Highlight, draw, and comment directly on documents

### 🤖 Chat with Your Documents
- **AI Assistant**: Ask questions about your uploaded content
- **Smart Memory**: Remembers previous conversations in each thread
- **Web Search**: Optionally include live internet results
- **Reasoning Display**: See how the AI thinks through problems

### 🎨 Easy to Use
- **Modern Interface**: Clean, intuitive design
- **Thread Organization**: Keep different topics separate
- **Customizable**: Adjust AI behavior per conversation
- **Private**: Everything runs locally on your machine

---

## 📖 How to Use

### Getting Started
1. **Create a Thread** - Use the sidebar to start a new conversation
2. **Add Content** - Upload PDFs or add webpage URLs
3. **Start Reading** - Click Play to hear documents aloud
4. **Ask Questions** - Type questions in the chat

### Reading & Audio
- **Play Controls**: Click Play or double-click any sentence to start
- **Voice Settings**: Choose different voices and adjust speed (0.5x-2.0x)
- **Auto-Scroll**: Document follows along automatically

### Chatting with AI
- **Select Model**: Choose your preferred AI model
- **Internet Search**: Toggle to include live web results
- **View Reasoning**: Expand panels to see AI thinking process
- **Semantic Memory**: See which past conversations were used

### Customization
- **Thread Settings**: Click ⚙️ to adjust AI behavior
- **System Role**: Change the AI's persona
- **Tool Instructions**: Guide how AI uses different tools
- **Custom Instructions**: Add extra directions

---

## 🔧 Technical Details

<details>
<summary>🏗️ Architecture & Services</summary>

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      Docker Compose                                         │
├─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────────┤
│    Frontend     │   RAG Service   │  Browser Capture│   PostgreSQL    │      Weaviate       │
│   (Next.js)     │    (FastAPI)    │   (Selenium)    │   (Primary DB)  │   (Vector DB)       │
│   Port: 3000    │   Port: 8000    │   Port: 7800    │   Port: 5432    │   Port: 8080        │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────────┘
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
| **Frontend** | 3000 | Next.js React app with PDF viewer, chat UI, thread management, and TTS |
| **RAG Service** | 8000 | FastAPI server for PDF processing, document indexing, AI chat, thread/message/file management |
| **Browser Capture** | 7800 | Selenium-based service for interactive webpage capture and PDF conversion |
| **PostgreSQL** | 5432 | Primary database for threads, messages, files, settings, and annotations |
| **Weaviate** | 8080 | Vector database for semantic and memory search |
| **DMR/Ollama/LMStudio** | 12434 | Local LLM server (external, user-provided) |

</details>

<details>
<summary>🤖 Advanced AI Features</summary>

### Multi-Agent Architecture
- **Orchestrator Agent**: LangGraph-powered agent that plans, selects tools, and synthesizes answers
- **Intent Agent** (optional): Pre-processes questions to improve query clarity and search precision
- **Tool-Calling**: Dynamic tool selection including document search, memory recall, web search, and clarification
- **Configurable Iterations**: Control tool-call rounds with forced final answer to prevent infinite loops

### Reasoning & Thinking Support
- **Multi-Provider Extraction**: Supports reasoning traces from Claude, OpenAI o-series, DeepSeek, QwQ, Qwen3-Thinking
- **Database Storage**: Reasoning traces persisted alongside answers in PostgreSQL
- **UI Display**: Expandable reasoning panels in chat bubbles for transparent AI thinking

### RAG & Semantic Memory
- **Thread-Scoped Collections**: Each thread has isolated vector collections in Weaviate
- **Multi-Source Retrieval**: Simultaneous search across PDFs, webpages, and past conversations
- **Semantic Recollection**: UI highlights which past messages were used in current answers
- **Context Management**: Intelligent token budgeting for optimal LLM context window usage

</details>

<details>
<summary>🛠️ Technology Stack</summary>

### RAG Service
| Technology | Purpose |
|------------|---------|
| **FastAPI** | Web framework |
| **LangChain** | LLM/Embedding integration |
| **LangGraph** | Stateful multi-agent workflow |
| **Weaviate Client** | Vector database operations |
| **SQLModel** | ORM built on SQLAlchemy |
| **SQLAlchemy** | Async database operations |
| **Alembic** | Database migration management |
| **asyncpg** | Async PostgreSQL driver |

### Browser Capture Service
| Technology | Purpose |
|------------|---------|
| **Selenium** | WebDriver automation |
| **Brave Browser** | Headless browser rendering |
| **WeasyPrint** | PDF conversion fallback |
| **FastAPI** | Service API framework |

### Frontend
| Technology | Purpose |
|------------|---------|
| **Next.js** | React framework |
| **Material-UI (MUI)** | UI components (v7) |
| **EmbedPDF** | PDF rendering with annotations |
| **react-markdown** | Chat message rendering |
| **React Query** | Async state management |

</details>

<details>
<summary>📁 Project Structure</summary>

```
askpdf/
├── docker-compose.yml          # Multi-service orchestration
├── run_tests.sh               # Comprehensive test runner
├── browser_capture/           # Selenium-based webpage capture service
├── rag_service/               # FastAPI backend with AI, RAG, and database
│   ├── app/
│   │   ├── api/               # REST API route handlers
│   │   ├── agent/             # Multi-agent AI system
│   │   ├── db/                # PostgreSQL data layer
│   │   ├── services/          # Business logic services
│   │   └── rag/               # RAG core logic
│   └── tests/                 # Comprehensive test suite
└── frontend/                  # Next.js React application
    ├── src/
    │   ├── components/        # UI components
    │   ├── hooks/             # React hooks
    │   └── lib/               # Utility functions
    └── package.json
```

</details>

<details>
<summary>⚙️ Configuration & Environment</summary>

### Environment Variables

Environment variables are now managed using a `.env` file for better security and maintainability. The system uses two approaches:

1. **`.env` file** - For user-configurable settings (models, database URLs, behavior settings)
2. **`docker-compose.yml`** - For service-specific configuration (networking, basic service settings)

#### Quick Setup

1. **Copy the example file**:
   ```bash
   cp .env.example .env
   ```

2. **Configure your LLM provider** in `.env`:
   ```env
   # Choose your LLM provider
   LLM_API_URL=http://host.docker.internal:1234/v1  # LMStudio
   # LLM_API_URL=http://host.docker.internal:11434   # Ollama  
   # LLM_API_URL=http://host.docker.internal:12434   # Docker Model Runner
   ```

3. **Review other settings** in `.env` and adjust as needed for your use case.

#### Environment Variables Reference

##### User-Configurable Variables (.env file)

**LLM Configuration**
| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_URL` | (none) | External LLM server URL (Docker Model Runner/Ollama/LMStudio) |

**Model Configuration**
| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-m3` | Single local embedding model to use |
| `LOCAL_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Single local reranker model to use |
| `EMBEDDING_DEVICE` | `cpu` | Device for embedding models (cpu/cuda/mps) |
| `RERANKER_DEVICE` | `cpu` | Device for reranker models (cpu/cuda/mps) |

**AI Behavior & Limits**
| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_TOKEN_BUDGET` | `8192` | Context window size for AI responses |
| `DEFAULT_MAX_ITERATIONS` | `10` | Maximum tool-call rounds for AI reasoning |
| `MIN_MAX_ITERATIONS` | `1` | Minimum allowed iterations |
| `MAX_MAX_ITERATIONS` | `30` | Maximum allowed iterations |
| `MAX_CUSTOM_INSTRUCTIONS_CHARS` | `2000` | Maximum custom instruction length |
| `MAX_SYSTEM_ROLE_CHARS` | `500` | Maximum system role description length |
| `MAX_TOOL_INSTRUCTION_CHARS` | `500` | Maximum tool instruction length |
| `INTENT_AGENT_MAX_ITERATIONS` | `1` | Maximum iterations for intent agent |
| `MAX_ITERATIONS_SUFFICIENT_COVERAGE` | `2` | Iteration bonus for sufficient coverage |
| `MAX_ITERATIONS_PROBABLY_SUFFICIENT_COVERAGE` | `4` | Iteration bonus for probable sufficient coverage |
| `WEB_SEARCH_ITERATION_BONUS` | `2` | Extra iterations when web search is enabled |

**Document Processing (Docling)**
| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_DO_OCR` | `True` | Enable OCR for scanned images (preserves digital text) |
| `DOCLING_DO_TABLE_STRUCTURE` | `True` | Extract table structure from documents |
| `DOCLING_TABLE_MODE` | `ACCURATE` | Table extraction mode (FAST/ACCURATE) |
| `DOCLING_FORCE_FULL_PAGE_OCR` | `False` | Force full-page OCR (keep false for digital PDFs) |
| `DOCLING_DO_FORMULA_ENRICHMENT` | `False` | Enable mathematical formula extraction |

**Database Configuration**
| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@postgresql:5432/askpdf` | PostgreSQL connection string |
| `TEST_DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@postgresql:5432/test_askpdf` | Test database connection string |
| `POSTGRES_POOL_SIZE` | `10` | Database connection pool size |
| `POSTGRES_MAX_OVERFLOW` | `20` | Maximum additional connections beyond pool size |

##### Service-Specific Variables (docker-compose.yml)

**Frontend Service**
| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | RAG service API URL for frontend communication |

**RAG Service - Core Configuration**
| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `WEAVIATE_URL` | `http://weaviate:8080` | Weaviate vector database endpoint |
| `WEAVIATE_HYBRID_ALPHA` | `0.7` | Hybrid search balance (0.0=pure vector, 1.0=pure keyword) |
| `CAPTURE_SERVICE_URL` | `http://browser-capture:8080` | Browser capture service endpoint |

### Setup Instructions

1. **Initial Setup**: Copy `.env.example` to `.env` and configure your settings:
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

2. **Apply Changes**: After modifying environment variables, restart the services:
   ```bash
   docker-compose down
   docker-compose up --build
   ```

### Model Requirements

You need a **chat model with tool calling support** and an **embedding model**:

| Runtime | Chat model example | Embedding model example |
|---------|-------------------|------------------------|
| DMR | `ai/qwen3:latest` | `ai/nomic-embed-text-v1.5:latest` |
| Ollama | `llama3.2` | `nomic-embed-text` |
| LMStudio | `google/gemma-3-12b` | `text-embedding-embeddinggemma-300m-qat`|

</details>

<details>
<summary>📝 API Reference</summary>

### Key Endpoints

#### Chat & Threads
- `POST /api/threads` - Create new thread
- `POST /api/threads/{thread_id}/chat` - Chat with documents
- `PUT /api/threads/{thread_id}/settings` - Update thread settings
- `GET /api/threads/{thread_id}/messages` - List messages

#### Files & Documents
- `POST /api/threads/{thread_id}/files/upload` - Upload PDF
- `GET /api/threads/{thread_id}/files/{file_hash}` - Get file data
- `GET /api/threads/{thread_id}/files/{file_hash}/status` - Check processing status

#### Models & Health
- `GET /api/models` - List available models
- `GET /api/health/chat-model/{model}` - Check chat model health
- `GET /api/health/embed-model/{model}` - Check embedding model health

</details>

<details>
<summary>🧭 Parallel Supabase Migration</summary>

askPDF can run self-hosted Supabase in parallel with the existing SQLModel/Postgres database. The current Postgres service remains the source of truth while Supabase is populated, validated, and enabled behind feature flags.

### Start Supabase Services

```bash
docker compose -f docker-compose.yml -f docker-compose.supabase.yml up -d
```

### Run the Docker Migrator

```bash
docker compose -f docker-compose.yml -f docker-compose.supabase.yml -f docker-compose.migration.yml run --rm --build db-migrate --dry-run
docker compose -f docker-compose.yml -f docker-compose.supabase.yml -f docker-compose.migration.yml run --rm --build db-migrate --migrate --allow-existing-target
docker compose -f docker-compose.yml -f docker-compose.supabase.yml -f docker-compose.migration.yml run --rm --build db-migrate --validate-only
```

Useful migration options:

- `--resume` continues into a populated target.
- `--allow-existing-target` permits idempotent upserts into a non-empty target.
- `--backup` records backup intent in the run metadata; create your database backup before running the container.
- `--verbose` prints full diagnostics on failure.

### Feature Flags

Keep these disabled until migration validation passes. Once DB parity, live dual-write, and Storage parity have passed, the rollout-ready local settings are:

```env
RAG_SUPABASE_DUAL_WRITE=true
RAG_SUPABASE_PARITY_CHECK=true
NEXT_PUBLIC_USE_SUPABASE_THREADS=true
NEXT_PUBLIC_USE_SUPABASE_MESSAGES=true
NEXT_PUBLIC_USE_SUPABASE_FILES=true
NEXT_PUBLIC_USE_SUPABASE_REALTIME=true
NEXT_PUBLIC_USE_SUPABASE_STORAGE=true
```

FastAPI still owns chat, uploads, browser capture, parsing/indexing, model health, and Weaviate cleanup. Supabase-backed frontend reads and Storage URLs are enabled only through the flags above, with FastAPI download fallback retained.

With Storage enabled, newly uploaded PDFs are mirrored to Supabase Storage in the background after the primary FastAPI upload succeeds. The migrator's `--validate-only` mode checks both table parity and Storage parity: local PDFs must have matching `files.storage_bucket/storage_path` metadata and corresponding `storage.objects` rows.

Recommended live smoke test:

1. Create a thread.
2. Upload a PDF.
3. Ask a question.
4. Refresh the browser and open/download the PDF.
5. Run the three-file `--validate-only` command above and confirm all table mismatches are `0` and Storage reports no missing metadata or objects.

### Supabase Soak Mode

Use soak mode after migration validation and the live smoke test have passed. Keep primary Postgres/FastAPI as the source of truth, leave Supabase reads enabled, and use the app normally for a short period before removing any legacy read paths.

Soak flags:

```env
RAG_SUPABASE_DUAL_WRITE=true
RAG_SUPABASE_PARITY_CHECK=true
NEXT_PUBLIC_USE_SUPABASE_THREADS=true
NEXT_PUBLIC_USE_SUPABASE_MESSAGES=true
NEXT_PUBLIC_USE_SUPABASE_FILES=true
NEXT_PUBLIC_USE_SUPABASE_REALTIME=true
NEXT_PUBLIC_USE_SUPABASE_STORAGE=true
```

Daily soak checks:

1. Create, rename, and delete a thread.
2. Upload a PDF, ask a question, refresh, and reopen the PDF.
3. Attach the same PDF to two threads, remove it from one, and confirm the other thread still opens it.
4. Remove the last thread reference to a test PDF and confirm normal app behavior continues.
5. Run `--validate-only` and confirm table mismatches are `0`, `missing_metadata_count` is `0`, and `missing_object_count` is `0`.

The app also exposes `GET /api/health/supabase` for a non-secret rollout status summary. The thread sidebar shows this status as a small cloud icon; click it to refresh the health probe.

Quick read rollback:

```env
NEXT_PUBLIC_USE_SUPABASE_THREADS=false
NEXT_PUBLIC_USE_SUPABASE_MESSAGES=false
NEXT_PUBLIC_USE_SUPABASE_FILES=false
NEXT_PUBLIC_USE_SUPABASE_REALTIME=false
NEXT_PUBLIC_USE_SUPABASE_STORAGE=false
```

Leave `RAG_SUPABASE_DUAL_WRITE=true` during rollback when possible so Supabase keeps receiving primary writes while frontend reads fall back to FastAPI. Disable `RAG_SUPABASE_DUAL_WRITE` only if the Supabase database or network path is causing noisy backend failures that need to be isolated.

</details>

<details>
<summary>🧪 Testing</summary>

### Running Tests

```bash
./run_tests.sh [options]
```

### Test Options
- `--verbose` - Verbose output
- `--file <file>` - Run specific test file
- `--coverage` - Run with coverage report
- `--db-tests` - Run PostgreSQL database tests
- `--api` - Run API endpoint tests

### Test Categories
- **Database Tests**: PostgreSQL operations, models, repositories
- **API Tests**: Endpoint testing, integration tests
- **Parsing Tests**: PDF processing with Docling and pdfplumber

</details>

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project uses the following third-party technologies:
- [Kokoro](https://github.com/hexgrad/kokoro) - Text-to-speech model
- [spaCy](https://spacy.io/) - Natural language processing
- [LangChain](https://langchain.com/) - LLM framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Stateful AI workflows
- [Weaviate](https://weaviate.tech/) - Vector database
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Next.js](https://nextjs.org/) - React framework

## 🙏 Acknowledgments

- **hexgrad** for the amazing Kokoro-82M model
- **spaCy** for robust NLP capabilities
- **LangChain** team for the excellent LLM framework
- **Weaviate** for the powerful vector database
- The open-source community for all the amazing tools

## 📧 Contact

For questions, issues, or suggestions, please open an issue on the [GitHub repository](https://github.com/raghu13590/askpdf).
