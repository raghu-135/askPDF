"""Unit tests for embedding projection helpers."""

import pytest
import torch

from app.services.embedding_projection_service import (
    compute_projection_edges,
    merge_embedding_family_points,
    project_embeddings_3d,
    resolve_embedding_source_families,
)


def test_project_embeddings_3d_empty():
    assert project_embeddings_3d([]) == []


def test_project_embeddings_3d_single_vector():
    assert project_embeddings_3d([[1.0, 2.0, 3.0]]) == [(0.0, 0.0, 0.0)]


def test_project_embeddings_3d_two_vectors():
    result = project_embeddings_3d([[1.0, 0.0], [0.0, 1.0]])
    assert len(result) == 2
    for coords in result:
        assert len(coords) == 3
        assert all(-1.0 <= value <= 1.0 for value in coords)


def test_project_embeddings_3d_many_vectors_are_bounded():
    vectors = torch.randn(32, 16).tolist()
    result = project_embeddings_3d(vectors)
    assert len(result) == 32
    for coords in result:
        assert len(coords) == 3
        assert all(-1.0 <= value <= 1.0 for value in coords)


def test_project_embeddings_3d_preserves_pairwise_structure():
    vectors = [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [2.0, 1.0, 4.0, 3.0]]
    first = torch.tensor(project_embeddings_3d(vectors), dtype=torch.float32)
    second = torch.tensor(project_embeddings_3d(vectors), dtype=torch.float32)
    first_distances = torch.cdist(first, first)
    second_distances = torch.cdist(second, second)
    assert torch.allclose(first_distances, second_distances, atol=1e-5)


def test_compute_projection_edges_generates_sequence_edges():
    points = [
        {"id": "p0", "file_hash": "f1", "chunk_id": 0},
        {"id": "p1", "file_hash": "f1", "chunk_id": 1},
        {"id": "p2", "file_hash": "f2", "chunk_id": 0},
    ]
    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ]
    edges = compute_projection_edges(points, vectors, similarity_threshold=0.99)
    seq_edges = [e for e in edges if e["kind"] == "sequence"]
    assert len(seq_edges) == 1
    assert seq_edges[0]["source"] == "p0"
    assert seq_edges[0]["target"] == "p1"
    assert seq_edges[0]["label"] == "next"


def test_compute_projection_edges_generates_similarity_edges():
    points = [
        {"id": "p0", "file_hash": "f1", "chunk_id": 0},
        {"id": "p1", "file_hash": "f2", "chunk_id": 0},
    ]
    # Identical vectors -> cosine similarity = 1.0
    vectors = [
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ]
    edges = compute_projection_edges(points, vectors, similarity_threshold=0.8)
    sim_edges = [e for e in edges if e["kind"] == "similarity"]
    assert len(sim_edges) == 1
    assert sim_edges[0]["source"] in ("p0", "p1")
    assert sim_edges[0]["target"] in ("p0", "p1")
    assert sim_edges[0]["score"] == 1.0


def test_compute_projection_edges_skips_empty_file_hash_sequence_groups():
    points = [
        {"id": "p0", "file_hash": "", "chunk_id": 0},
        {"id": "p1", "file_hash": "", "chunk_id": 1},
    ]
    edges = compute_projection_edges(points, [[1.0], [0.0]], similarity_threshold=0.99)
    assert [edge["kind"] for edge in edges] == []


def test_resolve_embedding_source_families_defaults_to_all():
    assert resolve_embedding_source_families(None) == (
        "documents",
        "chat",
        "web_search",
        "memory",
    )
    assert resolve_embedding_source_families("chat") == ("chat",)


def test_resolve_embedding_source_families_rejects_unknown():
    with pytest.raises(ValueError, match="source_family"):
        resolve_embedding_source_families("images")


def test_merge_embedding_family_points_interleaves_and_caps():
    merged, truncated = merge_embedding_family_points(
        {
            "documents": [{"id": "d0"}, {"id": "d1"}],
            "chat": [{"id": "c0"}, {"id": "c1"}],
        },
        family_order=("documents", "chat"),
        limit=3,
    )
    assert [point["id"] for point in merged] == ["d0", "c0", "d1"]
    assert truncated is True


def test_merge_embedding_family_points_not_truncated_when_all_fit():
    merged, truncated = merge_embedding_family_points(
        {
            "documents": [{"id": "d0"}],
            "chat": [{"id": "c0"}],
        },
        family_order=("documents", "chat"),
        limit=10,
    )
    assert [point["id"] for point in merged] == ["d0", "c0"]
    assert truncated is False

