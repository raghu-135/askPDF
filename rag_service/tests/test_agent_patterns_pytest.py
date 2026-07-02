import os
import uuid
import logging
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent_patterns.router_runtime import handle_router_rag_chat
from app.agent_patterns.graph import TemplateCompiler
from app.agent_patterns.repository import AgentPatternRepository
from app.agent_patterns.service import AgentRunService
from app.agent_patterns.templates import (
    ROUTER_RAG_AGENT_ID,
    ROUTER_RAG_AGENT_VERSION,
    builtin_router_rag_spec,
)
from app.agent_patterns.validator import TemplateResolver, TemplateValidationError, TemplateValidator
from app.db.models_sqlmodel import ChatTurn


SQLMODEL_AVAILABLE = bool(os.getenv("TEST_DATABASE_URL"))


class TestRouterRagTemplateValidator:
    @pytest.mark.parametrize(
        "mutate, expected",
        [
            (lambda spec: spec.update({"pattern_type": "simple_rag_agent"}), "pattern_type must be router_rag_agent"),
            (lambda spec: spec["config"].update({"surprise": True}), "unknown config keys: surprise"),
            (lambda spec: spec["config"].update({"allowed_tool_ids": ["not_a_tool"]}), "unknown allowed_tool_ids: not_a_tool"),
            (lambda spec: spec["config"].update({"max_iterations": 999}), "max_iterations must be between"),
        ],
    )
    def test_rejects_invalid_router_rag_specs(self, mutate, expected):
        spec = builtin_router_rag_spec()
        mutate(spec)

        with pytest.raises(TemplateValidationError) as exc:
            TemplateValidator().validate(spec)

        assert expected in str(exc.value)

    def test_resolver_freezes_thread_and_request_overrides(self):
        resolved = TemplateResolver().resolve(
            builtin_router_rag_spec(),
            thread_settings={"max_iterations": 3, "use_reranker": False},
            request_overrides={"use_web_search": True},
        )

        assert resolved["config"]["max_iterations"] == 3
        assert resolved["config"]["use_reranker"] is False
        assert resolved["config"]["use_web_search"] is True

    def test_accepts_builtin_router_rag_spec(self):
        result = TemplateValidator().validate(builtin_router_rag_spec())

        assert result == {"valid": True, "errors": []}

    def test_rejects_router_rag_graph_topology_changes(self):
        spec = builtin_router_rag_spec()
        spec["config"]["graph"]["nodes"].append({"id": "surprise", "type": "retrieval_worker"})

        with pytest.raises(TemplateValidationError) as exc:
            TemplateValidator().validate(spec)

        assert "graph nodes must match" in str(exc.value)

    def test_compiles_builtin_router_rag_spec(self):
        graph = TemplateCompiler().compile(builtin_router_rag_spec())

        assert graph is not None


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel test database is not configured")
class TestAgentPatternRepository:
    @pytest_asyncio.fixture
    async def repo(self, engine):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        async with session_factory() as repo_session:
            yield AgentPatternRepository(repo_session)

    @pytest.mark.asyncio
    async def test_seed_builtin_router_rag_template_is_idempotent(self, repo):
        await repo.seed_builtin_templates()
        await repo.seed_builtin_templates()

        templates = await repo.list_templates()
        template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)

        assert {template.id for template in templates} == {ROUTER_RAG_AGENT_ID}
        assert template.current_version_id == version.id
        assert version.version == ROUTER_RAG_AGENT_VERSION
        assert version.validation_result_json == {"valid": True, "errors": []}

    @pytest.mark.asyncio
    async def test_run_lifecycle_persists_resolved_spec(self, repo, sample_thread):
        await repo.seed_builtin_templates()
        template, version = await repo.get_template_with_current_version(ROUTER_RAG_AGENT_ID)

        run = await repo.create_run(
            thread_id=sample_thread.id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json={"pattern_type": ROUTER_RAG_AGENT_ID},
        )
        completed = await repo.complete_run(
            run.id,
            status="completed",
            metrics_json={"duration_ms": 12.5},
        )

        assert completed.status == "completed"
        assert completed.metrics_json == {"duration_ms": 12.5}
        assert completed.resolved_spec_json == {"pattern_type": ROUTER_RAG_AGENT_ID}

    @pytest.mark.asyncio
    async def test_unsupported_simple_rag_template_is_not_exposed(self, repo):
        await repo.seed_builtin_templates()

        template, version = await repo.get_template_with_current_version("simple_rag_agent")

        assert template is None
        assert version is None


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel test database is not configured")
class TestAgentRunService:
    @pytest.mark.asyncio
    async def test_run_thread_chat_falls_back_to_router_for_unsupported_simple_setting(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        captured_context = {}

        async with session_factory() as repo_session:
            repo = AgentPatternRepository(repo_session)
            await repo.seed_builtin_templates()

            async def fake_get_thread_settings(_thread_id):
                return {"agent_pattern": {"template_id": "simple_rag_agent"}}

            async def fake_handle_router_rag_chat(_thread_id, _req, _embed_model, *, resolved_spec, agent_run_context):
                captured_context.update(agent_run_context or {})
                return {
                    "answer": "router fallback",
                    "document_sources": [{"id": "doc"}],
                    "web_sources": [],
                    "used_chat_ids": [],
                    "clarification_options": None,
                    "route": "direct",
                    "node_events": [],
                    **agent_run_context,
                }

            monkeypatch.setattr("app.agent_patterns.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr("app.agent_patterns.router_runtime.handle_router_rag_chat", fake_handle_router_rag_chat)

            req = SimpleNamespace(
                question="What is this about?",
                llm_model="test-llm",
                use_intent_agent=False,
                use_web_search=False,
                use_reranker=True,
                max_iterations=1,
                system_role_override="",
                tool_instructions_override={},
                custom_instructions_override="",
            )
            result = await AgentRunService(repository=repo).run_thread_chat(
                sample_thread.id,
                req,
                sample_thread.embed_model,
            )

            run = await repo.get_run(result["agent_run_id"])

        assert result["agent_pattern_id"] == ROUTER_RAG_AGENT_ID
        assert result["agent_pattern_version"] == ROUTER_RAG_AGENT_VERSION
        assert captured_context["agent_run_id"] == result["agent_run_id"]
        assert run.status == "completed"
        assert run.metrics_json["document_source_count"] == 1
        assert run.resolved_spec_json["pattern_type"] == ROUTER_RAG_AGENT_ID

    @pytest.mark.asyncio
    async def test_run_thread_chat_defaults_to_router_rag(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        async with session_factory() as repo_session:
            repo = AgentPatternRepository(repo_session)
            await repo.seed_builtin_templates()

            async def fake_get_thread_settings(_thread_id):
                return {}

            async def fake_handle_router_rag_chat(_thread_id, _req, _embed_model, *, resolved_spec, agent_run_context):
                return {
                    "answer": "router default",
                    "document_sources": [],
                    "web_sources": [],
                    "used_chat_ids": [],
                    "clarification_options": None,
                    "route": "direct",
                    "node_events": [],
                    **agent_run_context,
                }

            monkeypatch.setattr("app.agent_patterns.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr("app.agent_patterns.router_runtime.handle_router_rag_chat", fake_handle_router_rag_chat)

            req = SimpleNamespace(
                question="What is this about?",
                llm_model="test-llm",
                use_intent_agent=False,
                use_web_search=False,
                use_reranker=True,
                max_iterations=1,
                system_role_override="",
                tool_instructions_override={},
                custom_instructions_override="",
            )
            result = await AgentRunService(repository=repo).run_thread_chat(
                sample_thread.id,
                req,
                sample_thread.embed_model,
            )

        assert result["agent_pattern_id"] == ROUTER_RAG_AGENT_ID

    @pytest.mark.asyncio
    async def test_run_thread_chat_uses_router_rag_when_selected(self, engine, sample_thread, monkeypatch):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        captured_spec = {}

        async with session_factory() as repo_session:
            repo = AgentPatternRepository(repo_session)
            await repo.seed_builtin_templates()

            async def fake_get_thread_settings(_thread_id):
                return {"agent_pattern": {"template_id": ROUTER_RAG_AGENT_ID}}

            async def fake_handle_router_rag_chat(_thread_id, _req, _embed_model, *, resolved_spec, agent_run_context):
                captured_spec.update(resolved_spec)
                return {
                    "answer": "router ok",
                    "document_sources": [],
                    "web_sources": [],
                    "used_chat_ids": [],
                    "clarification_options": None,
                    "route": "direct",
                    "node_events": [{"node": "router"}],
                    **agent_run_context,
                }

            monkeypatch.setattr("app.agent_patterns.service.get_thread_settings", fake_get_thread_settings)
            monkeypatch.setattr("app.agent_patterns.router_runtime.handle_router_rag_chat", fake_handle_router_rag_chat)

            req = SimpleNamespace(
                question="What is this about?",
                llm_model="test-llm",
                use_intent_agent=False,
                use_web_search=False,
                use_reranker=True,
                max_iterations=1,
                system_role_override="",
                tool_instructions_override={},
                custom_instructions_override="",
            )
            result = await AgentRunService(repository=repo).run_thread_chat(
                sample_thread.id,
                req,
                sample_thread.embed_model,
            )

            run = await repo.get_run(result["agent_run_id"])

        assert result["agent_pattern_id"] == ROUTER_RAG_AGENT_ID
        assert captured_spec["pattern_type"] == ROUTER_RAG_AGENT_ID
        assert run.status == "completed"
        assert run.metrics_json["route"] == "direct"
        assert run.metrics_json["node_event_count"] == 1


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel test database is not configured")
class TestRouterRagRuntime:
    @pytest.mark.asyncio
    async def test_handle_router_rag_chat_runs_compiled_direct_route_and_persists_turn(
        self,
        engine,
        sample_thread,
        monkeypatch,
        caplog,
    ):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        class FakeEmbeddingModel:
            async def aembed_query(self, query):
                return [0.1, 0.2, 0.3]

        class FakeVectorDb:
            async def search_knowledge_sources(self, **kwargs):
                return [
                    {
                        "text": "DiffusionBlocks is about modular diffusion model research.",
                        "file_hash": "file-1",
                        "chunk_id": 1,
                        "score": 0.9,
                    }
                ]

        class FakeLlm:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, messages):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(
                        content='{"route":"direct","reason":"prefetched context is sufficient","clarification_options":null}'
                    )
                return SimpleNamespace(content="DiffusionBlocks focuses on modular diffusion model research.")

        fake_llm = FakeLlm()

        async def fake_get_recent_messages(_thread_id, limit):
            return []

        async def fake_get_thread_shape(_thread_id):
            return {
                "total_qa_pairs": 0,
                "total_qa_chars": 0,
                "documents": {
                    "file-1": {
                        "file_name": "diffusionblocks.pdf",
                        "source_type": "pdf",
                        "document_available_in_thread_at": "2026-07-02T00:00:00Z",
                        "chunk_count": 1,
                        "total_chars": 128,
                        "word_count": 18,
                        "page_count": 1,
                        "sentence_count": 1,
                    }
                },
            }

        async def fake_fetch_semantic_history(**kwargs):
            return "", []

        async def fake_get_document_metadata_lookup(_thread_id):
            return {
                "file-1": {
                    "file_name": "diffusionblocks.pdf",
                    "source_type": "pdf",
                    "document_available_in_thread_at": "2026-07-02T00:00:00Z",
                }
            }

        def fake_group_document_chunks(chunks, lookup, char_budget=None):
            return (
                "[Source: PDF: diffusionblocks.pdf]\nDiffusionBlocks is about modular diffusion model research.",
                [{"file_hash": "file-1", "file_name": "diffusionblocks.pdf"}],
            )

        async def fake_index_chat_memory_for_thread(**kwargs):
            return {"memory_compact_text": "Q/A compact"}

        async def fake_update_message_context_compact(_turn_id, _compact_text):
            return None

        async def fake_increment_qa_stats(_thread_id, _qa_chars):
            return None

        async def fake_create_chat_turn(
            *,
            thread_id,
            question,
            answer,
            rewritten_question=None,
            status="completed",
            reasoning="",
            reasoning_available=False,
            reasoning_format="none",
            web_sources=None,
            document_sources=None,
            used_chat_ids=None,
            clarification_options=None,
            error=None,
            metadata=None,
        ):
            turn = ChatTurn(
                id=str(uuid.uuid4()),
                thread_id=thread_id,
                status=status,
                payload={
                    "question": question,
                    "rewritten_question": rewritten_question,
                    "answer": answer,
                    "reasoning": reasoning,
                    "reasoning_available": reasoning_available,
                    "reasoning_format": reasoning_format,
                    "web_sources": web_sources or [],
                    "document_sources": document_sources or [],
                    "used_chat_ids": used_chat_ids or [],
                    "clarification_options": clarification_options,
                    "error": error,
                    "metadata": metadata or {},
                },
            )
            async with session_factory() as write_session:
                write_session.add(turn)
                await write_session.commit()
                await write_session.refresh(turn)
            return turn

        monkeypatch.setattr("app.rag.chat_service.get_embedding_model", lambda _name: FakeEmbeddingModel())
        monkeypatch.setattr("app.rag.chat_service.get_recent_messages", fake_get_recent_messages)
        monkeypatch.setattr("app.rag.chat_service.get_thread_shape", fake_get_thread_shape)
        monkeypatch.setattr("app.rag.chat_service.fetch_semantic_history", fake_fetch_semantic_history)
        monkeypatch.setattr("app.rag.chat_service.get_document_metadata_lookup", fake_get_document_metadata_lookup)
        monkeypatch.setattr("app.rag.chat_service.group_document_chunks", fake_group_document_chunks)
        monkeypatch.setattr("app.db.vector.get_vector_db", lambda: FakeVectorDb())
        monkeypatch.setattr("app.agent_patterns.graph.get_llm", lambda _name: fake_llm)
        monkeypatch.setattr("app.agent_patterns.router_runtime.index_chat_memory_for_thread", fake_index_chat_memory_for_thread)
        monkeypatch.setattr("app.agent_patterns.router_runtime.create_chat_turn", fake_create_chat_turn)
        monkeypatch.setattr("app.agent_patterns.router_runtime.update_message_context_compact", fake_update_message_context_compact)
        monkeypatch.setattr("app.agent_patterns.router_runtime.increment_qa_stats", fake_increment_qa_stats)

        req = SimpleNamespace(
            question="What is this document about?",
            llm_model="test-llm",
            use_web_search=False,
            use_reranker=False,
            context_window=8192,
            system_role_override="",
            tool_instructions_override={},
            custom_instructions_override="",
            client_timezone="America/Chicago",
            client_locale="en-US",
            client_now_iso="2026-07-02T12:00:00.000Z",
        )

        caplog.set_level(logging.INFO, logger="app.agent_patterns")
        result = await handle_router_rag_chat(
            sample_thread.id,
            req,
            sample_thread.embed_model,
            resolved_spec=builtin_router_rag_spec(),
            agent_run_context={
                "agent_run_id": "run-1",
                "agent_pattern_id": ROUTER_RAG_AGENT_ID,
                "agent_pattern_version": ROUTER_RAG_AGENT_VERSION,
            },
        )

        async with session_factory() as check_session:
            turn = await check_session.get(ChatTurn, result["user_message_id"].split(":")[0])

        assert result["answer"] == "DiffusionBlocks focuses on modular diffusion model research."
        assert result["route"] == "direct"
        assert [event["node"] for event in result["node_events"]] == [
            "context_loader",
            "router",
            "direct_answer",
            "finalizer",
        ]
        assert result["agent_run_id"] == "run-1"
        assert turn is not None
        assert turn.status == "completed"
        assert turn.payload["metadata"]["agent_run_id"] == "run-1"
        assert turn.payload["metadata"]["agent_route"] == "direct"

        log_text = "\n".join(record.getMessage() for record in caplog.records)
        assert "Router RAG run started | run_id=run-1" in log_text
        assert "Router RAG run completed | run_id=run-1" in log_text
        for node in ("context_loader", "router", "direct_answer", "finalizer"):
            assert f"Router RAG node completed | run_id=run-1" in log_text
            assert f"node={node}" in log_text
        assert "route=direct" in log_text

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "route, expected_nodes, expected_status",
        [
            ("document", ["context_loader", "router", "retrieval_worker", "synthesizer", "finalizer"], "completed"),
            ("memory", ["context_loader", "router", "memory_worker", "synthesizer", "finalizer"], "completed"),
            ("timeline", ["context_loader", "router", "timeline_worker", "synthesizer", "finalizer"], "completed"),
            ("web", ["context_loader", "router", "web_worker", "synthesizer", "finalizer"], "completed"),
            ("clarify", ["context_loader", "router", "finalizer"], "clarification"),
        ],
    )
    async def test_handle_router_rag_chat_covers_compiled_routes(
        self,
        engine,
        sample_thread,
        monkeypatch,
        caplog,
        route,
        expected_nodes,
        expected_status,
    ):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        class FakeLlm:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, messages):
                self.calls += 1
                if self.calls == 1:
                    options = '["Which uploaded document?","Which previous answer?"]' if route == "clarify" else "null"
                    return SimpleNamespace(
                        content=f'{{"route":"{route}","reason":"test route","clarification_options":{options}}}'
                    )
                return SimpleNamespace(content=f"Final answer from {route} route.")

        class FakeTool:
            def __init__(self, payload):
                self.payload = payload

            async def ainvoke(self, _args, config=None):
                return self.payload

        async def fake_prefetch_context(**kwargs):
            return {
                "recent_history_text": "",
                "semantic_history_text": "Prior answer about DiffusionBlocks.",
                "document_evidence_text": "Document evidence about DiffusionBlocks.",
                "web_evidence_text": "",
                "stats": {"total_messages": 0, "estimated_history_tokens": 0},
                "documents": [{"file_name": "diffusionblocks.pdf", "file_hash": "file-1"}],
                "document_sources": [],
                "web_sources": [],
                "used_chat_ids": [],
            }

        async def fake_index_chat_memory_for_thread(**kwargs):
            return {}

        async def fake_update_message_context_compact(_turn_id, _compact_text):
            return None

        async def fake_increment_qa_stats(_thread_id, _qa_chars):
            return None

        async def fake_create_chat_turn(
            *,
            thread_id,
            question,
            answer,
            rewritten_question=None,
            status="completed",
            reasoning="",
            reasoning_available=False,
            reasoning_format="none",
            web_sources=None,
            document_sources=None,
            used_chat_ids=None,
            clarification_options=None,
            error=None,
            metadata=None,
        ):
            turn = ChatTurn(
                id=str(uuid.uuid4()),
                thread_id=thread_id,
                status=status,
                payload={
                    "question": question,
                    "rewritten_question": rewritten_question,
                    "answer": answer,
                    "reasoning": reasoning,
                    "reasoning_available": reasoning_available,
                    "reasoning_format": reasoning_format,
                    "web_sources": web_sources or [],
                    "document_sources": document_sources or [],
                    "used_chat_ids": used_chat_ids or [],
                    "clarification_options": clarification_options,
                    "error": error,
                    "metadata": metadata or {},
                },
            )
            async with session_factory() as write_session:
                write_session.add(turn)
                await write_session.commit()
                await write_session.refresh(turn)
            return turn

        document_payload = {
            "content": "Document worker evidence.",
            "__document_sources__": [{"file_hash": "file-1", "file_name": "diffusionblocks.pdf"}],
        }
        memory_payload = {
            "content": "Memory worker evidence.",
            "__used_chat_ids__": ["turn-1"],
        }
        timeline_payload = {
            "content": "Timeline worker evidence.",
            "__timeline_events__": [{"timeline_event_type": "document_added", "timeline_event_at": "2026-07-01T00:00:00Z"}],
        }
        web_payload = {
            "content": "Web worker evidence.",
            "__web_sources__": [{"url": "https://example.com", "title": "Example"}],
        }
        fake_llm = FakeLlm()

        monkeypatch.setattr("app.agent_patterns.graph.prefetch_context", fake_prefetch_context)
        monkeypatch.setattr("app.agent_patterns.graph.get_llm", lambda _name: fake_llm)
        monkeypatch.setattr("app.agent_patterns.graph.search_documents", FakeTool(document_payload))
        monkeypatch.setattr("app.agent_patterns.graph.search_conversation_history", FakeTool(memory_payload))
        monkeypatch.setattr("app.agent_patterns.graph.search_thread_timeline", FakeTool(timeline_payload))
        monkeypatch.setattr("app.agent_patterns.graph.search_web", FakeTool(web_payload))
        monkeypatch.setattr("app.agent_patterns.router_runtime.index_chat_memory_for_thread", fake_index_chat_memory_for_thread)
        monkeypatch.setattr("app.agent_patterns.router_runtime.create_chat_turn", fake_create_chat_turn)
        monkeypatch.setattr("app.agent_patterns.router_runtime.update_message_context_compact", fake_update_message_context_compact)
        monkeypatch.setattr("app.agent_patterns.router_runtime.increment_qa_stats", fake_increment_qa_stats)

        req = SimpleNamespace(
            question="Route coverage?",
            llm_model="test-llm",
            use_web_search=route == "web",
            use_reranker=False,
            context_window=8192,
            system_role_override="",
            tool_instructions_override={},
            custom_instructions_override="",
            client_timezone=None,
            client_locale=None,
            client_now_iso=None,
        )

        caplog.set_level(logging.INFO, logger="app.agent_patterns")
        result = await handle_router_rag_chat(
            sample_thread.id,
            req,
            sample_thread.embed_model,
            resolved_spec=builtin_router_rag_spec(),
            agent_run_context={
                "agent_run_id": f"run-{route}",
                "agent_pattern_id": ROUTER_RAG_AGENT_ID,
                "agent_pattern_version": ROUTER_RAG_AGENT_VERSION,
            },
        )

        async with session_factory() as check_session:
            turn = await check_session.get(ChatTurn, result["user_message_id"].split(":")[0])

        assert result["route"] == route
        assert [event["node"] for event in result["node_events"]] == expected_nodes
        assert turn is not None
        assert turn.status == expected_status
        assert turn.payload["metadata"]["agent_route"] == route
        if route == "document":
            assert result["document_sources"] == [{"file_hash": "file-1", "file_name": "diffusionblocks.pdf"}]
            assert result["answer"] == "Final answer from document route."
        elif route == "memory":
            assert result["used_chat_ids"] == ["turn-1"]
            assert result["answer"] == "Final answer from memory route."
        elif route == "timeline":
            assert result["answer"] == "Final answer from timeline route."
        elif route == "web":
            assert result["web_sources"] == [{"url": "https://example.com", "title": "Example"}]
            assert result["answer"] == "Final answer from web route."
        else:
            assert result["clarification_options"] == ["Which uploaded document?", "Which previous answer?"]
            assert result["answer"].startswith("I need a bit more clarification.")

        log_text = "\n".join(record.getMessage() for record in caplog.records)
        assert f"Router RAG run completed | run_id=run-{route}" in log_text
        assert f"route={route}" in log_text
        for node in expected_nodes:
            assert f"node={node}" in log_text


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel test database is not configured")
class TestAgentPatternApi:
    def test_list_and_get_builtin_agent_pattern(self, api_client):
        listed = api_client.get("/api/agent-patterns")
        assert listed.status_code == 200
        assert {item["id"] for item in listed.json()["agent_patterns"]} == {ROUTER_RAG_AGENT_ID}

        detail = api_client.get(f"/api/agent-patterns/{ROUTER_RAG_AGENT_ID}")
        assert detail.status_code == 200
        payload = detail.json()
        assert payload["agent_pattern"]["id"] == ROUTER_RAG_AGENT_ID
        assert payload["current_version"]["version"] == ROUTER_RAG_AGENT_VERSION

        stale_detail = api_client.get("/api/agent-patterns/simple_rag_agent")
        assert stale_detail.status_code == 404

    def test_validate_agent_pattern_endpoint(self, api_client):
        valid = api_client.post(
            "/api/agent-patterns/validate",
            json={"spec": builtin_router_rag_spec()},
        )
        invalid_spec = builtin_router_rag_spec()
        invalid_spec["config"]["allowed_tool_ids"] = ["mystery_tool"]
        stale_spec = builtin_router_rag_spec()
        stale_spec["pattern_type"] = "simple_rag_agent"
        invalid = api_client.post(
            "/api/agent-patterns/validate",
            json={"spec": invalid_spec},
        )
        stale = api_client.post(
            "/api/agent-patterns/validate",
            json={"spec": stale_spec},
        )

        assert valid.status_code == 200
        assert valid.json() == {"valid": True, "errors": []}
        assert invalid.status_code == 200
        assert invalid.json()["valid"] is False
        assert stale.status_code == 200
        assert stale.json()["valid"] is False
