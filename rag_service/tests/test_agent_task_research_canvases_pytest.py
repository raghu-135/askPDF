import uuid

import pytest

from app.models.canvas import RESEARCH_CANVAS_ARTIFACT_KIND
from app.services.agent_task_research_canvases import list_research_canvas_refs_for_task_ids


@pytest.mark.asyncio
async def test_list_research_canvas_refs_groups_by_task(test_session_maker, sample_thread):
    from app.db.models_sqlmodel import AgentTask, AgentTaskArtifact, AgentRun, AgentWorkflow

    workflow_id = str(uuid.uuid4())
    task_a = str(uuid.uuid4())
    task_b = str(uuid.uuid4())
    run_a = str(uuid.uuid4())
    canvas_a = str(uuid.uuid4())
    async with test_session_maker() as session:
        async with session.begin():
            session.add(AgentWorkflow(
                id=workflow_id,
                name=workflow_id,
                framework="langgraph",
                builder_id="langgraph_graph",
                spec_json={},
            ))
            await session.flush()
            session.add(AgentTask(
                id=task_a,
                thread_id=sample_thread.id,
                project_id=sample_thread.project_id,
                workflow_id=workflow_id,
                objective="A",
                objective_hash=task_a,
                create_idempotency_key=task_a,
                status="completed",
                version=1,
            ))
            session.add(AgentTask(
                id=task_b,
                thread_id=sample_thread.id,
                project_id=sample_thread.project_id,
                workflow_id=workflow_id,
                objective="B",
                objective_hash=task_b,
                create_idempotency_key=task_b,
                status="completed",
                version=1,
            ))
            await session.flush()
            session.add(AgentRun(
                id=run_a,
                task_id=task_a,
                thread_id=sample_thread.id,
                workflow_id=workflow_id,
                framework="langgraph",
                builder_id="langgraph_graph",
                task_attempt=2,
                status="completed",
            ))
            await session.flush()
            session.add(AgentTaskArtifact(
                id=canvas_a,
                task_id=task_a,
                agent_run_id=run_a,
                thread_id=sample_thread.id,
                ownership_key=f"task:{task_a}:canvas:{canvas_a}",
                kind=RESEARCH_CANVAS_ARTIFACT_KIND,
                object_key=f"threads/{sample_thread.id}/artifacts/{canvas_a}",
                media_type="application/vnd.askpdf.canvas+json",
                byte_size=10,
                sha256="abc",
                validity="valid",
                summary_json={"title": "Employers timeline"},
            ))

    refs = await list_research_canvas_refs_for_task_ids(sample_thread.id, [task_a, task_b])
    assert refs[task_b] == []
    assert len(refs[task_a]) == 1
    assert refs[task_a][0]["id"] == canvas_a
    assert refs[task_a][0]["title"] == "Employers timeline"
    assert refs[task_a][0]["attempt"] == 2
    assert refs[task_a][0]["run_id"] == run_a
