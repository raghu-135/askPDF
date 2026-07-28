from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.db import (
    assign_thread_to_project,
    create_project,
    create_thread,
    ensure_default_project,
    get_project,
    list_projects,
    update_project,
)
from app.models.requests import ProjectCreateRequest, ProjectUpdateRequest, ThreadCreateRequest
from app.time_utils import iso_utc_z
from app.services.memory_policy import merge_project_settings_json


router = APIRouter(tags=["projects"])


def _project_payload(project) -> Dict[str, Any]:
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "embedding_model": project.embedding_model,
        "settings_json": merge_project_settings_json(project.settings_json),
        "created_at": iso_utc_z(project.created_at) if project.created_at else None,
        "updated_at": iso_utc_z(project.updated_at) if project.updated_at else None,
    }


def _thread_payload(thread) -> Dict[str, Any]:
    return {
        "id": thread.id,
        "project_id": thread.project_id,
        "name": thread.name,
        "embedding_model": thread.embedding_model,
        "settings": thread.settings if isinstance(thread.settings, dict) else {},
        "thread_metadata": thread.thread_metadata if isinstance(thread.thread_metadata, dict) else {},
        "created_at": iso_utc_z(thread.created_at) if thread.created_at else None,
    }


@router.get("/projects")
async def list_projects_endpoint():
    await ensure_default_project()
    projects = await list_projects()
    return {"projects": [_project_payload(project) for project in projects]}


@router.post("/projects")
async def create_project_endpoint(req: ProjectCreateRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Project name is required")
    embedding_model = req.embedding_model.strip()
    if not embedding_model:
        raise HTTPException(status_code=400, detail="embedding_model is required")
    project = await create_project(
        name=name,
        embedding_model=embedding_model,
        description=req.description,
        settings_json=req.settings_json,
    )
    return _project_payload(project)


@router.get("/projects/{project_id}")
async def get_project_endpoint(project_id: str):
    project = await get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return _project_payload(project)


@router.put("/projects/{project_id}")
async def update_project_endpoint(project_id: str, req: ProjectUpdateRequest):
    project = await update_project(
        project_id,
        name=req.name.strip() if isinstance(req.name, str) else None,
        description=req.description,
        settings_json=req.settings_json,
    )
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return _project_payload(project)


@router.post("/projects/{project_id}/threads")
async def create_project_thread_endpoint(project_id: str, req: ThreadCreateRequest):
    project = await get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    thread = await create_thread(req.name, project_id)
    return _thread_payload(thread)


@router.put("/projects/{project_id}/threads/{thread_id}")
async def assign_project_thread_endpoint(project_id: str, thread_id: str):
    try:
        thread = await assign_thread_to_project(thread_id, project_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if thread is None:
        raise HTTPException(status_code=404, detail="Project or thread not found")
    return _thread_payload(thread)
