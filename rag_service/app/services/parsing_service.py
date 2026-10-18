import io
import os
import logging
import json
from dataclasses import dataclass, field
from typing import Any, Optional

# Docling 2.118 compiles layout (RT-DETR) models by default; eager inference is
# equivalent for extraction quality and avoids torch._dynamo graph-break noise.
os.environ.setdefault("DOCLING_INFERENCE_COMPILE_TORCH_MODELS", "false")

from docling.document_converter import DocumentConverter, DocumentStream, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import CoordOrigin
from app.services.document_extraction_contract import extraction_configuration as _lightweight_extraction_configuration
from app.services.document_pipeline import sentence_pipeline_identity

logger = logging.getLogger(__name__)

# Initialize Docling converter as a module-level singleton
_pipeline_options = PdfPipelineOptions()


def _get_env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() == "true"


def _get_env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


# Tuneable via environment variables for accuracy vs speed
_pipeline_options.do_ocr = _get_env_bool("DOCLING_DO_OCR", False)
_pipeline_options.do_table_structure = _get_env_bool("DOCLING_DO_TABLE_STRUCTURE", True)
_pipeline_options.do_formula_enrichment = _get_env_bool("DOCLING_DO_FORMULA_ENRICHMENT", True)
_pipeline_options.generate_parsed_pages = True

# Table structure mode (FAST or ACCURATE)
_table_mode = _get_env_str("DOCLING_TABLE_MODE", "FAST").upper()
if _table_mode == "ACCURATE":
    _pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
else:
    _pipeline_options.table_structure_options.mode = TableFormerMode.FAST

# OCR options
if _pipeline_options.do_ocr:
    _pipeline_options.ocr_options.force_full_page_ocr = _get_env_bool("DOCLING_FORCE_FULL_PAGE_OCR", False)

_docling_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options)
    }
)


@dataclass
class DoclingParseResult:
    """Docling document plus per-page word cells for deterministic alignment."""

    document: Any
    page_sizes: dict[int, tuple[float, float]] = field(default_factory=dict)
    word_cells_by_page: dict[int, list[dict[str, Any]]] = field(default_factory=dict)


def extraction_configuration() -> dict[str, object]:
    """Return every parsing input that can change the canonical document."""
    configuration = _lightweight_extraction_configuration(sentence_model=sentence_pipeline_identity())
    configuration.update(
        {
            "do_ocr": bool(_pipeline_options.do_ocr),
            "do_table_structure": bool(_pipeline_options.do_table_structure),
            "do_formula_enrichment": bool(_pipeline_options.do_formula_enrichment),
            "generate_parsed_pages": bool(_pipeline_options.generate_parsed_pages),
            "table_mode": str(
                getattr(_pipeline_options.table_structure_options.mode, "value", _pipeline_options.table_structure_options.mode)
            ).upper(),
            "force_full_page_ocr": bool(
                getattr(_pipeline_options.ocr_options, "force_full_page_ocr", False)
            ),
        }
    )
    return configuration


def _word_cells_for_page(page: Any) -> list[dict[str, Any]]:
    parsed = getattr(page, "parsed_page", None)
    if parsed is None:
        return []
    page_no = int(getattr(page, "page_no", 0) or 0)
    page_height = float(getattr(page.size, "height", 0) or 0)
    cells: list[dict[str, Any]] = []
    for index, cell in enumerate(getattr(parsed, "word_cells", None) or []):
        rect = cell.rect.to_bounding_box()
        if rect.coord_origin != CoordOrigin.TOPLEFT:
            rect = rect.to_top_left_origin(page_height)
        cells.append(
            {
                "index": index,
                "page": page_no,
                "text": str(getattr(cell, "text", "") or ""),
                "x0": float(rect.l),
                "x1": float(rect.r),
                "top": float(rect.t),
                "bottom": float(rect.b),
            }
        )
    return cells


def parse_with_docling(
    data: bytes,
    filename: str,
    debug_output_dir: Optional[str] = None,
) -> DoclingParseResult | None:
    """
    Parse PDF with Docling to extract structure, bounding boxes, and word cells.
    """
    if not filename:
        return None

    try:
        source = DocumentStream(name=filename, stream=io.BytesIO(data))
        docling_result = _docling_converter.convert(source)
        document = docling_result.document

        page_sizes: dict[int, tuple[float, float]] = {}
        word_cells_by_page: dict[int, list[dict[str, Any]]] = {}
        for page in docling_result.pages or []:
            page_no = int(getattr(page, "page_no", 0) or 0)
            if page_no <= 0:
                continue
            width = float(getattr(page.size, "width", 0) or 0)
            height = float(getattr(page.size, "height", 0) or 0)
            if width > 0 and height > 0:
                page_sizes[page_no] = (width, height)
            word_cells_by_page[page_no] = _word_cells_for_page(page)

        if debug_output_dir:
            os.makedirs(debug_output_dir, exist_ok=True)
            test_file_json = os.path.join(debug_output_dir, f"docling_raw_output_{filename}_dict.json")
            try:
                doc_dict = document.export_to_dict()
                with open(test_file_json, "w") as f:
                    json.dump(doc_dict, f, indent=2)
                logger.info(f"Raw Docling output written to {test_file_json}")
            except Exception as e:
                logger.warning(f"Failed to write Docling output: {e}")

        return DoclingParseResult(
            document=document,
            page_sizes=page_sizes,
            word_cells_by_page=word_cells_by_page,
        )
    except Exception as e:
        logger.error("Docling conversion failed", exc_info=True)
        raise RuntimeError("Docling conversion failed") from e
