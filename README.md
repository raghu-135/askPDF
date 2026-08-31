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
docker compose up --build
```

### 4. Use It!
- **Open**: http://localhost:3000
- **Upload PDFs** or add webpages
- **Click Play** or select PDF text and use the read-aloud icon to hear documents aloud
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
- **Play Controls**: Click Play, or select PDF text and click the read-aloud icon in the selection menu
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
│   Port: 3000    │   Port: 8000    │   Port: 8090    │   Port: 5432    │   Port: 8080        │
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
| **RAG Service** | 8000 | FastAPI server for PDF processing, indexing, chat, and the integrated durable agent-task worker |
| **Browser Capture** | 8090 | Selenium-based service for interactive webpage capture and PDF conversion |
| **PostgreSQL** | 5432 | Primary database for threads, messages, files, settings, and annotations |
| **Weaviate** | 8080 | Vector database for semantic and memory search |
| **DMR/Ollama/LMStudio** | 12434 | Local LLM server (external, user-provided) |

The current deployment runs `rag-service` as one Uvicorn process. Its integrated
agent-task worker shares the service's PostgreSQL pool and uses database leases
and checkpoints for restart recovery. Do not enable multiple Uvicorn/Gunicorn
worker processes until agent execution is extracted into its planned dedicated
service; each server process would otherwise start another task worker.

</details>

<details>
<summary>🤖 Advanced AI Features</summary>

### Multi-Agent Architecture
- **Agent Workflow Runtime**: LangGraph-powered Router RAG and Plan-and-Execute RAG workflows with persisted run metadata
- **Human-in-the-Loop Gates**: Optional web-search approval and resumable checkpoints for agent runs awaiting review
- **Tool Contracts**: First-party tool contracts for document search, memory recall, timeline search, web search, and clarification
- **Debug Traces**: Run-level trace payloads for inspecting routes, node execution, tool calls, warnings, and errors

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
| `HF_TOKEN` | (optional) | Hugging Face token for higher model-download rate limits |
| `EMBEDDING_DEVICE` | `cpu` | Device for embedding models (cpu/cuda/mps) |
| `RERANKER_DEVICE` | `cpu` | Device for reranker models (cpu/cuda/mps) |

**AI Behavior & Limits**
| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_TOKEN_BUDGET` | `8192` | Context window size for AI responses |
| `REPLANS_LIMIT` | `3` | Maximum allowed replans |
| `MAX_CUSTOM_INSTRUCTIONS_CHARS` | `2000` | Maximum custom instruction length |
| `MAX_SYSTEM_ROLE_CHARS` | `500` | Maximum system role description length |
| `MAX_TOOL_INSTRUCTION_CHARS` | `500` | Maximum tool instruction length |
| `INTENT_AGENT_MAX_ITERATIONS` | `1` | Maximum replans for intent agent |
| `MAX_ITERATIONS_SUFFICIENT_COVERAGE` | `2` | Iteration bonus for sufficient coverage |
| `MAX_ITERATIONS_PROBABLY_SUFFICIENT_COVERAGE` | `4` | Iteration bonus for probable sufficient coverage |
| `WEB_SEARCH_ITERATION_BONUS` | `2` | Extra replans when web search is enabled |

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
| `NEXT_PUBLIC_API_URL` | Required | Public RAG service URL baked into the frontend at build time; the frontend refuses to start or build when it is missing or blank |

**RAG Service - Core Configuration**
| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `WEAVIATE_URL` | `http://weaviate:8080` | Weaviate vector database endpoint |
| `WEAVIATE_HYBRID_ALPHA` | `0.7` | Hybrid search balance (0.0=pure vector, 1.0=pure keyword) |
| `CAPTURE_SERVICE_URL` | `http://browser-capture:8080` | Browser capture service endpoint |
| `LANGGRAPH_RUNTIME_URL` | `http://langgraph-runtime:8100` | Required internal URL for the external LangGraph runtime |
| `ASKPDF_CONTENT_ROOT` | `/static` | Backend-only shared-volume root for PDFs and Deep Research artifacts |

**Agent Runtime Operations**
- Local development runs the control plane and `langgraph-runtime` together through Compose, or points `LANGGRAPH_RUNTIME_URL` at a separately launched runtime. The control plane has no in-process LangGraph mode.
- Checkpoint configuration and credentials belong only to `langgraph-runtime`; the runtime fails closed when durable checkpoint storage is unavailable.
- Built-in workflow JSON files are loaded and seeded automatically at startup. Their runtime features, limits, and profiles are authoritative; no workflow feature flags are required.
- The visible web-search approval toggle is a UI/thread-settings convenience shim. New agent runs normalize it into `config.hitl_policy.gates.web_approval_gate`, and the reusable backend contract is `hitl_policy.gates`, where gates can target any actionable graph node by `node_id` or `node_type` and run before or after that node.
- Agent debug traces redact secret-like keys such as tokens, API keys, cookies, and authorization headers, and bound long preview/raw values before persisting.
- Stale running-run cleanup and pending-interrupt expiration are separate operations. Cleanup for stale `running` rows must not mark `awaiting_human` runs failed; pending review rows should transition through interrupt expiration.
- Runtime checkpoint administration is performed from the `langgraph-runtime` image and never from the control plane.
- The Hermes adapter implements the production API contract pinned to NousResearch/hermes-agent commit `bdd0a79c6a0ebc2344d5d6913c70bd89fa59c894`. Definitions resolve deterministically into managed profiles; credentials remain environment-owned. The bundled gateway journal is still single-worker/single-replica and requires a shared transactional store before horizontal scaling.

The default Compose stack builds the pinned Hermes API and its askPDF adapter.
Set `API_SERVER_KEY` in `.env`, then start the application normally. Hermes uses
the model selected for each askPDF thread and the existing OpenAI-compatible
`LLM_API_URL`; no separate Hermes model setting is required:

```bash
docker compose up --build
```

Hermes uses `/health`; the adapter uses `/readyz`, which also verifies MCP.

### Setup Instructions

1. **Initial Setup**: Copy `.env.example` to `.env` and configure your settings:
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

2. **Apply Changes**: After modifying environment variables, restart the services:
   ```bash
   docker compose down
   docker compose up --build
   ```

### Docker Portability

The Compose setup builds the frontend with `npm ci` inside Docker and runs the production Next.js standalone server, so users do not need Node or npm installed locally.

For frontend development with hot reload, add the dev override:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
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
<summary>🧪 Testing</summary>

### Running Tests

```bash
./run_tests.sh [options]
```

The test runner is Docker-native. `run_tests.sh` starts an isolated
`askpdf-test` Compose project with its own PostgreSQL, Weaviate, network, and
volumes, so macOS, Linux, Windows with Docker/WSL, and GitHub Actions all use
the same test environment. The normal app stack can keep running while tests
run because the test services do not publish host ports.

You can also run the test container directly:

```bash
docker compose -p askpdf-test -f docker-compose.test.yml run --rm --build test-runner
docker compose -p askpdf-test -f docker-compose.test.yml run --rm --build test-runner --api
docker compose -p askpdf-test -f docker-compose.test.yml run --rm --build test-runner --group db
```

Set `ASKPDF_TEST_PROJECT_NAME` to override the isolated Compose project name.
Set `ASKPDF_KEEP_TEST_CONTAINERS=1` to keep test containers and volumes after a
run for debugging.

### Test Options
- `--verbose` - Verbose output
- `--file <file>` - Run specific test file
- `--test <test>` - Run a specific test inside `--file`
- `--coverage` - Run with coverage report
- `--unit` - Run unit and mock-based tests
- `--db` / `--db-tests` / `--db-only` - Run PostgreSQL database tests
- `--api` - Run API endpoint tests
- `--integration` - Run integration tests
- `--agent-checkpoint` - Run the Postgres checkpoint/resume hardening test
- `--schema` - Run schema guardrail tests
- `--standalone` - Run standalone verification scripts
- `--all` / `--all-tests` - Run the full pytest suite plus standalone checks

### Test Categories
- **Database Tests**: PostgreSQL operations, models, repositories
- **API Tests**: Endpoint testing, integration tests
- **Parsing Tests**: PDF processing with Docling and pdfplumber

### CI and Merge Gates

GitHub Actions runs Docker build, the default Docker test runner, and a focused
Postgres checkpoint/resume hardening lane on pull requests and pushes to
`main`. To block merges unless CI passes, configure a branch ruleset in GitHub:

1. Go to **Settings → Rules → Rulesets**.
2. Create a ruleset for `main`.
3. Require pull requests before merging.
4. Require status checks to pass.
5. Select the `Docker build` and `Test suite` checks from the `CI` workflow.
6. Block force pushes and branch deletions.

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
# Hermes Deep Research runtime

Hermes is an opt-in engine for durable Deep Research tasks; standard chat workflows remain on LangGraph. The integration is pinned to `NousResearch/hermes-agent@bdd0a79c6a0ebc2344d5d6913c70bd89fa59c894` (Hermes config schema 37).

Hermes is opt-in through one switch. Set `COMPOSE_PROFILES=hermes`, `API_SERVER_KEY`, `OPENAI_API_KEY`, `HERMES_MCP_CONTEXT_SECRET` (at least 32 random characters), `HERMES_MODEL_CONTEXT_LENGTH`, and `HERMES_MODEL_PROVIDER`, then run `docker compose up -d`. The same `COMPOSE_PROFILES` value both starts the three Hermes services and advertises Hermes through rag-service; unset it to disable Hermes. Check `docker compose ps` and `curl -f http://localhost:8200/readyz`; stop the stack with `docker compose down`. The default stack does not build or start Hermes and needs no GitHub access or Hermes secrets. There is no production context-length default: Hermes startup and task creation remain unavailable when the value is missing, nonnumeric, or below 2,048. The pinned revision normally requires at least 64,000 tokens, but explicitly permits a smaller configured value for its first-class `lmstudio` provider. askPDF validates that compatibility rule, renders the exact deployment value into Hermes configuration, and freezes it into each new Hermes task; definitions contain no credentials.

LangGraph remains the default Deep Research engine. Select Hermes explicitly in the Deep Research workspace. The selected engine, model, context window, and workflow definition are frozen on the task and retained for retries and inspection.

Hermes Deep Research uses two reproducible managed profiles. The offline profile exposes thread shape, broad and focused document retrieval, conversation history, timeline, and durable-memory search. Tasks created with web mode `ask` or `on` use the external profile, which additionally exposes live web search, Wikipedia, Wikidata, arXiv, PubMed, Semantic Scholar, Stack Exchange, and Yahoo Finance News. For each run, askPDF derives a short-lived isolated profile and supplies its signed execution context through an MCP transport header; the model never receives or copies that credential. Legacy aliases and memory-curator mutation tools are not exposed, and the ordinary LangGraph MCP endpoint and contracts are unchanged. A Hermes report is published only after a permitted evidence tool returns nonempty evidence; otherwise the run fails with `required_evidence_unavailable` and remains retryable.

The pinned revision discovers MCP servers only at gateway startup. The Compose integration therefore mounts a revision-guarded askPDF compatibility hook that activates newly generated profile MCP servers during `/p/<profile>/v1/toolsets`, verifies the profile identity and complete tool surface before model execution, and retires only that run's MCP connection afterward. Remove the hook when the pinned upstream revision provides equivalent dynamic-profile lifecycle support.
