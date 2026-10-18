"""Serve-time table row highlight boxes from Docling cell geometry (display only)."""

from __future__ import annotations

from typing import Any, Mapping


def _positive_dimension(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _convert_bottomleft_to_topleft(
    bbox: tuple[float, float, float, float],
    page_height: float,
) -> tuple[float, float, float, float]:
    left, top, right, bottom = bbox
    return (left, page_height - top, right, page_height - bottom)


def _bbox_from_raw_cell(raw_cell: Mapping[str, Any], page_height: float) -> tuple[float, float, float, float] | None:
    bbox_data = raw_cell.get("bbox")
    if not isinstance(bbox_data, Mapping):
        return None
    try:
        left = float(bbox_data["l"])
        top = float(bbox_data["t"])
        right = float(bbox_data["r"])
        bottom = float(bbox_data["b"])
    except (KeyError, TypeError, ValueError):
        return None
    origin = str(bbox_data.get("coord_origin") or "TOPLEFT").upper()
    if origin == "BOTTOMLEFT":
        return _convert_bottomleft_to_topleft((left, top, right, bottom), page_height)
    return (left, top, right, bottom)


def _table_bounds_topleft(element: Mapping[str, Any], page_height: float) -> tuple[float, float, float, float] | None:
    for prov in element.get("provenance") or []:
        if not isinstance(prov, Mapping):
            continue
        bbox = _bbox_from_raw_cell({"bbox": prov.get("bbox")}, page_height)
        if bbox is not None:
            return bbox
    return None


def _normalized_raw_cells(element: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = element.get("raw") if isinstance(element.get("raw"), Mapping) else {}
    data = raw.get("data") if isinstance(raw.get("data"), Mapping) else {}
    raw_cells = data.get("table_cells")
    if not isinstance(raw_cells, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, raw_cell in enumerate(raw_cells):
        if not isinstance(raw_cell, Mapping):
            continue
        try:
            row_start = int(raw_cell["start_row_offset_idx"])
            row_end = int(raw_cell["end_row_offset_idx"])
            col_start = int(raw_cell["start_col_offset_idx"])
            col_end = int(raw_cell["end_col_offset_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        normalized.append({
            "index": index,
            "row_start": row_start,
            "row_end": row_end,
            "row_span": row_end - row_start,
            "col_start": col_start,
            "col_end": col_end,
            "text": str(raw_cell.get("text") or ""),
            "raw_cell": raw_cell,
        })
    return normalized


def _grid_row_from_table_row_id(sentence: Mapping[str, Any], element: Mapping[str, Any]) -> int | None:
    """Map projected data row N (table_row_id :row:N) to 0-based Docling grid row."""
    table_row_id = str(sentence.get("table_row_id") or "")
    if ":row:" not in table_row_id:
        return None
    suffix = table_row_id.rsplit(":row:", 1)[-1]
    try:
        data_index = int(suffix)
    except ValueError:
        return None
    structure = element.get("table_structure")
    if not isinstance(structure, Mapping):
        return None
    header_rows = set(structure.get("header_rows") or [])
    current_data_index = 0
    for grid_row_index, row in enumerate(structure.get("rows") or []):
        if grid_row_index in header_rows:
            continue
        values = [str(value).strip() for value in row]
        if not any(values):
            continue
        current_data_index += 1
        if current_data_index == data_index:
            return grid_row_index
    return None


def _grid_row_from_single_row_cells(cell_ids: list[str], cells: list[dict[str, Any]]) -> int | None:
    rows: set[int] = set()
    for cell_id in cell_ids:
        token = str(cell_id or "").split(":cell:")
        if len(token) != 2 or not token[1].isdigit():
            continue
        index = int(token[1])
        if 0 <= index < len(cells) and int(cells[index]["row_span"]) == 1:
            rows.add(int(cells[index]["row_start"]))
    return min(rows) if rows else None


def _resolve_grid_row(
    sentence: Mapping[str, Any],
    element: Mapping[str, Any],
    cell_ids: list[str],
    cells: list[dict[str, Any]],
) -> int | None:
    grid_row = _grid_row_from_table_row_id(sentence, element)
    if grid_row is not None:
        return grid_row
    return _grid_row_from_single_row_cells(cell_ids, cells)


def _row_highlight_box(
    cells: list[dict[str, Any]],
    grid_row: int,
    *,
    page: int,
    page_width: float,
    page_height: float,
    table_bounds: tuple[float, float, float, float],
) -> dict[str, Any] | None:
    xs: list[float] = []
    ys: list[float] = []
    table_x0, table_y0, table_x1, table_y1 = table_bounds
    for cell in cells:
        raw_cell = cell["raw_cell"]
        bbox = _bbox_from_raw_cell(raw_cell, page_height)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        row_start = int(cell["row_start"])
        row_end = int(cell["row_end"])
        row_span = int(cell["row_span"])
        if row_start == grid_row and row_span == 1:
            xs.extend([x0, x1])
            ys.extend([y0, y1])
        elif row_start <= grid_row < row_end and row_span > 1:
            xs.extend([x0, x1])
    if not ys and not xs:
        return None
    x0 = table_x0
    x1 = table_x1
    if not ys:
        y0, y1 = table_y0, table_y1
    else:
        y0, y1 = min(ys), max(ys)
    if x1 <= x0 or y1 <= y0:
        return None
    return {
        "page": page,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "page_width": page_width,
        "page_height": page_height,
    }


def _whole_table_box(
    element: Mapping[str, Any],
    *,
    page: int,
    page_width: float,
    page_height: float,
) -> dict[str, Any] | None:
    bounds = _table_bounds_topleft(element, page_height)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    return {
        "page": page,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "page_width": page_width,
        "page_height": page_height,
    }


def apply_table_row_highlight_boxes(
    sentences: list[dict[str, Any]],
    document_json: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Attach per-row highlight boxes for table sentences (does not change sentence text)."""
    if not sentences or not isinstance(document_json, Mapping):
        return sentences
    elements = {
        str(item.get("element_id")): item
        for item in document_json.get("elements") or []
        if isinstance(item, Mapping) and item.get("element_id")
    }
    table_cells_cache: dict[str, list[dict[str, Any]]] = {}
    enriched: list[dict[str, Any]] = []
    for sentence in sentences:
        row = dict(sentence)
        cell_ids = list(row.get("table_cell_ids") or [])
        if not cell_ids:
            enriched.append(row)
            continue
        element_ids = row.get("source_element_ids") or []
        element_id = str(element_ids[0]) if element_ids else ""
        element = elements.get(element_id)
        if element is None:
            enriched.append(row)
            continue
        page = int(row.get("page") or (row.get("pages") or [1])[0] or 1)
        page_width = _positive_dimension(row.get("page_width")) or 612.0
        page_height = _positive_dimension(row.get("page_height")) or 792.0
        if element_id not in table_cells_cache:
            table_cells_cache[element_id] = _normalized_raw_cells(element)
        cells = table_cells_cache[element_id]
        grid_row = _resolve_grid_row(row, element, cell_ids, cells)
        table_bounds = _table_bounds_topleft(element, page_height)
        row_box = None
        if grid_row is not None and table_bounds is not None:
            row_box = _row_highlight_box(
                cells,
                grid_row,
                page=page,
                page_width=page_width,
                page_height=page_height,
                table_bounds=table_bounds,
            )
        if row_box is None:
            row_box = _whole_table_box(element, page=page, page_width=page_width, page_height=page_height)
        if row_box is not None:
            row["bboxes"] = [row_box]
            row["bbox"] = row_box
            row["page"] = row_box["page"]
            row["page_width"] = row_box["page_width"]
            row["page_height"] = row_box["page_height"]
        enriched.append(row)
    return enriched
