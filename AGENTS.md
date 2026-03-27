# AI Coding Agents Guide for askPDF

## Architecture Overview
- **Microservices**: Frontend (Next.js), Backend (FastAPI), RAG Service (FastAPI), Qdrant (Vector DB). Services communicate via HTTP APIs (ports 3000, 8000, 8001, 6333).
- **Data Flow**: Upload PDFs/websites via Backend → index in RAG Service → store vectors in Qdrant → chat queries retrieve from Qdrant and external LLM.
- **Agent System**: LangGraph-powered orchestrator in `rag_service/agent.py` with tool-calling (e.g., document search, web search). Prompts in `rag_service/prompts/orchestrator/system.md`.
- **Persistence**: SQLite (`data/rag.db`) for threads/messages, Qdrant for vectors, local files for PDFs/audio.

## Key Workflows
- **Run Full Stack**: `docker-compose up` (requires `.env` with `LLM_API_URL` pointing to local LLM server like Ollama/DMR).
- **Develop Services**: Edit code in `backend/`, `rag_service/`, `frontend/`; volumes mount for hot-reload. Restart containers for Python changes.
- **Index Documents**: Backend extracts text/sentences via Docling, sends to RAG Service for embedding (default BGE-M3 local model).
- **Chat Flow**: Frontend calls RAG Service `/chat` → orchestrator plans tool calls → retrieves from Qdrant → synthesizes with LLM.

## Project Conventions
- **API Endpoints**: Prefixed `/api` in Backend/RAG. Use `httpx` for internal calls (e.g., Backend to RAG via `RAG_SERVICE_URL`).
- **Environment Config**: Service-specific env vars in `docker-compose.yml`; global `.env` for LLM URL.
- **Error Handling**: Raise `HTTPException` in FastAPI; log with print() or logging (no centralized logger).
- **Async Patterns**: Use `async def` for API routes; background tasks for indexing (e.g., `BackgroundTasks` in upload).
- **File Storage**: PDFs in `pdf_data` volume, audio in `/data/audio`, DB in `./data/rag.db`.
- **Model Selection**: Dynamic LLM/embedding model choice per thread; cached in localStorage.

## Integration Points
- **External LLM**: OpenAI-compatible API (e.g., Ollama at `http://host.docker.internal:11434`); tool-calling required for agents.
- **Web Search**: DuckDuckGo via `rag_service/agent_helpers.py`; results stored in DB/Qdrant.
- **TTS**: Kokoro-82M in Backend for sentence audio; synchronized playback in Frontend.

## References
- Core logic: `rag_service/rag.py` (indexing), `rag_service/chat_service.py` (chat handling).
- UI components: `frontend/src/components/` (e.g., `ChatInterface.tsx` for threads).
- Config: `docker-compose.yml` for services, `rag_service/Dockerfile` for dependencies.</content>
<parameter name="filePath">/Users/raghu.bommineni/Documents/Repositories/Personal/askPDF/AGENTS.md