import pytest
import pytest_asyncio

from app.db.repositories.message_repo_sqlmodel import MessageRepository


class TestCanvasApi:
    @pytest_asyncio.fixture
    async def client(self, async_api_client):
        yield async_api_client

    @staticmethod
    def spec(title="Research canvas"):
        return {
            "schema_version": 1,
            "title": title,
            "sections": [
                {
                    "title": "Findings",
                    "blocks": [
                        {"type": "stat", "value": "2", "label": "Sources"},
                        {
                            "type": "sources",
                            "citations": [
                                {
                                    "kind": "document",
                                    "label": "Attached PDF",
                                    "file_hash": "doc1",
                                    "sentence_id": 4,
                                }
                            ],
                        },
                    ],
                }
            ],
        }

    @pytest.mark.asyncio
    async def test_create_list_and_get_canvas(self, client):
        thread = await client.post("/api/threads", json={"name": "Canvas thread"})
        assert thread.status_code == 200, thread.text
        thread_id = thread.json()["id"]

        created = await client.post(
            f"/api/threads/{thread_id}/canvases",
            json={"spec": self.spec(), "idempotency_key": "canvas-key-0001"},
        )
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["title"] == "Research canvas"
        assert body["current"] is True
        canvas_id = body["id"]

        replay = await client.post(
            f"/api/threads/{thread_id}/canvases",
            json={"spec": self.spec("Ignored title"), "idempotency_key": "canvas-key-0001"},
        )
        assert replay.status_code == 200
        assert replay.json()["id"] == canvas_id
        assert replay.json()["title"] == "Research canvas"

        listed = await client.get(f"/api/threads/{thread_id}/canvases")
        assert listed.status_code == 200
        assert [item["id"] for item in listed.json()["canvases"]] == [canvas_id]

        fetched = await client.get(f"/api/threads/{thread_id}/canvases/{canvas_id}")
        assert fetched.status_code == 200
        assert fetched.json()["spec"]["sections"][0]["blocks"][0]["type"] == "stat"

        await client.delete(f"/api/threads/{thread_id}")

    @pytest.mark.asyncio
    async def test_supersede_and_message_projection(self, client):
        thread = await client.post("/api/threads", json={"name": "Canvas messages"})
        thread_id = thread.json()["id"]
        turn = await MessageRepository().create_turn(
            thread_id=thread_id,
            question="Compare the papers",
            answer="See the canvas.",
        )
        created = await client.post(
            f"/api/threads/{thread_id}/canvases",
            json={"spec": self.spec("First"), "chat_turn_id": turn.id},
        )
        assert created.status_code == 200, created.text
        first_id = created.json()["id"]

        revised = await client.post(
            f"/api/threads/{thread_id}/canvases",
            json={"spec": self.spec("Second"), "chat_turn_id": turn.id, "supersedes_id": first_id},
        )
        assert revised.status_code == 200, revised.text
        second_id = revised.json()["id"]

        current = await client.get(f"/api/threads/{thread_id}/canvases")
        assert [item["id"] for item in current.json()["canvases"]] == [second_id]

        history = await client.get(f"/api/threads/{thread_id}/canvases?current_only=false")
        assert {item["id"] for item in history.json()["canvases"]} == {first_id, second_id}

        messages = await client.get(f"/api/threads/{thread_id}/messages")
        assert messages.status_code == 200
        assistant = next(item for item in messages.json()["messages"] if item["role"] == "assistant")
        assert assistant["canvas_ref"] == {"id": second_id, "title": "Second"}

        conflict = await client.post(
            f"/api/threads/{thread_id}/canvases",
            json={"spec": self.spec("Third"), "supersedes_id": first_id},
        )
        assert conflict.status_code == 400

        await client.delete(f"/api/threads/{thread_id}")

    @pytest.mark.asyncio
    async def test_rejects_unknown_component_and_missing_thread(self, client):
        missing = await client.post("/api/threads/missing-thread/canvases", json={"spec": self.spec()})
        assert missing.status_code == 404

        thread = await client.post("/api/threads", json={"name": "Invalid canvas"})
        thread_id = thread.json()["id"]
        invalid = await client.post(
            f"/api/threads/{thread_id}/canvases",
            json={"spec": {"schema_version": 1, "title": "Bad", "sections": [{"title": "X", "blocks": [{"type": "html", "markup": "<script>"}]}]}},
        )
        assert invalid.status_code == 400
        await client.delete(f"/api/threads/{thread_id}")
