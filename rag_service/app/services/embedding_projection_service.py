"""Project high-dimensional embedding vectors into 3D coordinates for visualization."""

from __future__ import annotations

import torch


def project_embeddings_3d(vectors: list[list[float]]) -> list[tuple[float, float, float]]:
    """Project embedding vectors into normalized 3D coordinates using PyTorch PCA.

    Returns one (x, y, z) tuple per input vector, each bounded to [-1.0, 1.0].
    """
    if not vectors:
        return []

    if len(vectors) == 1:
        return [(0.0, 0.0, 0.0)]

    matrix = torch.tensor(vectors, dtype=torch.float32)
    centered = matrix - torch.mean(matrix, dim=0, keepdim=True)

    if len(vectors) == 2:
        _, _, components = torch.pca_lowrank(centered, q=2)
        projected = torch.matmul(centered, components[:, :2])
        padded = torch.cat(
            [projected, torch.zeros((projected.shape[0], 1), dtype=projected.dtype)],
            dim=1,
        )
    else:
        component_count = min(3, centered.shape[0], centered.shape[1])
        _, _, components = torch.pca_lowrank(centered, q=component_count)
        projected = torch.matmul(centered, components[:, :component_count])
        if projected.shape[1] < 3:
            padding = torch.zeros(
                (projected.shape[0], 3 - projected.shape[1]),
                dtype=projected.dtype,
            )
            padded = torch.cat([projected, padding], dim=1)
        else:
            padded = projected[:, :3]

    max_abs = torch.max(torch.abs(padded)).item()
    if max_abs <= 0:
        return [(0.0, 0.0, 0.0) for _ in vectors]

    normalized = padded / max_abs
    return [tuple(float(value) for value in row) for row in normalized.tolist()]


def compute_projection_edges(
    points: list[dict],
    vectors: list[list[float]],
    *,
    similarity_threshold: float = 0.70,
    max_similarity_edges_per_node: int = 3,
) -> list[dict]:
    """Generate sequential reading edges and semantic similarity edges between chunk points."""
    if not points:
        return []

    edges: list[dict] = []
    seen_edge_ids: set[str] = set()

    # 1. Sequential reading order edges per document: chunk[i] -> chunk[i+1]
    by_file: dict[str, list[dict]] = {}
    for point in points:
        file_hash = str(point.get("file_hash") or "")
        by_file.setdefault(file_hash, []).append(point)

    for file_chunks in by_file.values():
        sorted_chunks = sorted(file_chunks, key=lambda p: int(p.get("chunk_id") or 0))
        for index in range(len(sorted_chunks) - 1):
            src = sorted_chunks[index]
            dst = sorted_chunks[index + 1]
            src_id = str(src.get("id"))
            dst_id = str(dst.get("id"))
            edge_id = f"seq:{src_id}:{dst_id}"
            if edge_id not in seen_edge_ids:
                seen_edge_ids.add(edge_id)
                edges.append({
                    "id": edge_id,
                    "source": src_id,
                    "target": dst_id,
                    "label": "next",
                    "kind": "sequence",
                    "score": None,
                })

    # 2. Semantic similarity edges using cosine similarity on normalized vectors
    if len(points) > 1 and len(vectors) == len(points):
        v_tensor = torch.tensor(vectors, dtype=torch.float32)
        norms = torch.norm(v_tensor, p=2, dim=1, keepdim=True)
        # Avoid division by zero
        normalized_v = v_tensor / torch.clamp(norms, min=1e-8)
        # Pairwise cosine similarities (N x N)
        similarity_matrix = torch.matmul(normalized_v, normalized_v.T)

        for i, point_i in enumerate(points):
            src_id = str(point_i.get("id"))
            # Get similarity scores for other nodes
            scores_with_idx = [
                (j, float(similarity_matrix[i, j].item()))
                for j in range(len(points))
                if i != j and float(similarity_matrix[i, j].item()) >= similarity_threshold
            ]
            # Take top K highest similarity neighbors
            scores_with_idx.sort(key=lambda item: item[1], reverse=True)
            for j, score in scores_with_idx[:max_similarity_edges_per_node]:
                dst_id = str(points[j].get("id"))
                # To prevent duplicate bidirectional edges, sort IDs
                pair_key = tuple(sorted([src_id, dst_id]))
                edge_id = f"sim:{pair_key[0]}:{pair_key[1]}"
                if edge_id not in seen_edge_ids:
                    seen_edge_ids.add(edge_id)
                    edges.append({
                        "id": edge_id,
                        "source": src_id,
                        "target": dst_id,
                        "label": f"{score:.2f}",
                        "kind": "similarity",
                        "score": round(score, 3),
                    })

    return edges

