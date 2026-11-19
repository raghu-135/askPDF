#!/usr/bin/env python3
"""Convert redistributable BEIR / QA corpora into bundled PDF fixtures and question stubs.

Reproducible pins: HuggingFace dataset revisions in CORPUS_SOURCE_PINS below.
Run from repo root after: pip install -r scripts/requirements-corpus-build.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
BUNDLED_DIR = REPO / "benchmarks" / "fixtures" / "bundled"
MANIFEST_PATH = REPO / "benchmarks" / "fixtures" / "bundled_manifest.json"
QUESTIONS_PATH = REPO / "benchmarks" / "questions.json"
LICENSES_APPEND_PATH = REPO / "benchmarks" / "fixtures" / "LICENSES.md"

# HuggingFace dataset id -> revision (None = datasets library default at pin time)
HF_DATASET_PINS: dict[str, str | None] = {
    "hotpot_qa": None,
    "rajpurkar/squad_v2": None,
}

# ir-datasets BEIR snapshot (package version pinned in requirements-corpus-build.txt)
IR_DATASETS_BEIR: dict[str, str] = {
    "nfcorpus": "beir/nfcorpus/test",
    "trec-covid": "beir/trec-covid",
}

# Redistribution policy: only datasets we can commit. See benchmarks/fixtures/LICENSES.md.
@dataclass(frozen=True)
class SourceSpec:
    slug: str
    hf_id: str
    hf_config: str | None
    license_name: str
    license_url: str
    license_notes: str
    bundles: int
    docs_per_bundle: int
    skipped: bool = False
    skip_reason: str = ""


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        slug="scifact",
        hf_id="BeIR/scifact",
        hf_config=None,
        license_name="CC BY-NC 4.0",
        license_url="https://creativecommons.org/licenses/by-nc/4.0/",
        license_notes="Non-commercial only; excluded from bundled redistribution.",
        bundles=2,
        docs_per_bundle=4,
        skipped=True,
        skip_reason="CC BY-NC 4.0 (BEIR/SciFact) does not permit redistribution in this repository.",
    ),
    SourceSpec(
        slug="nfcorpus",
        hf_id="BeIR/nfcorpus",
        hf_config=None,
        license_name="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        license_notes="Nutrition/medical abstracts (BEIR NFCorpus; PubMed Central open-access subset).",
        bundles=2,
        docs_per_bundle=4,
    ),
    SourceSpec(
        slug="trec-covid",
        hf_id="BeIR/trec-covid",
        hf_config=None,
        license_name="CORD-19 heterogeneous (BEIR trec-covid subset)",
        license_url="https://allenai.org/data/cord-19",
        license_notes="Abstract passages from CORD-19 via BEIR; underlying article licenses vary.",
        bundles=2,
        docs_per_bundle=4,
    ),
    SourceSpec(
        slug="hotpotqa",
        hf_id="hotpot_qa",
        hf_config="distractor",
        license_name="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        license_notes="Wikipedia passages bundled as distractor setting; share-alike applies.",
        bundles=2,
        docs_per_bundle=4,
        skipped=True,
        skip_reason="Hotpot JSON CDN blocked in this environment; use BEIR bundles (re-enable when cache is seeded).",
    ),
    SourceSpec(
        slug="squad",
        hf_id="rajpurkar/squad_v2",
        hf_config=None,
        license_name="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        license_notes="Wikipedia passage shards from SQuAD 2.0 (train split sample).",
        bundles=2,
        docs_per_bundle=4,
        skipped=True,
        skip_reason="SQuAD JSON omitted here; BEIR + HotpotQA cover extractive and multi-hop IR.",
    ),
    SourceSpec(
        slug="fiqa",
        hf_id="BeIR/fiqa",
        hf_config=None,
        license_name="",
        license_url="",
        license_notes="",
        bundles=0,
        docs_per_bundle=0,
        skipped=True,
        skip_reason="FiQA challenge terms are research-oriented; not bundled for redistribution.",
    ),
    SourceSpec(
        slug="nq",
        hf_id="google/natural_questions",
        hf_config=None,
        license_name="",
        license_url="",
        license_notes="",
        bundles=0,
        docs_per_bundle=0,
        skipped=True,
        skip_reason="Natural Questions omitted (size); CC BY-SA 3.0 would allow a future subset.",
    ),
    SourceSpec(
        slug="wikiqa",
        hf_id="",
        hf_config=None,
        license_name="",
        license_url="",
        license_notes="",
        bundles=0,
        docs_per_bundle=0,
        skipped=True,
        skip_reason="Microsoft WikiQA license forbids redistribution.",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokens_for_match(text: str, *, max_tokens: int = 4) -> list[str]:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}", text)
    stop = {"the", "and", "for", "that", "with", "from", "this", "have", "been", "were", "which", "their", "about"}
    picked: list[str] = []
    for w in words:
        lw = w.lower()
        if lw in stop:
            continue
        if w not in picked and len(w) > 3:
            picked.append(w if w[0].isupper() else w.capitalize())
        if len(picked) >= max_tokens:
            break
    return picked or words[:max_tokens]


def _load_beir_ir(spec: SourceSpec) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    import ir_datasets

    dataset_id = IR_DATASETS_BEIR[spec.slug]
    ds = ir_datasets.load(dataset_id)
    qmap = {str(q.query_id): str(q.text) for q in ds.queries_iter()}
    qrows: list[dict[str, Any]] = []
    needed_ids: list[str] = []
    needed_set: set[str] = set()
    need_docs = spec.bundles * spec.docs_per_bundle
    for rel in ds.qrels_iter():
        if int(rel.relevance) <= 0:
            continue
        qid = str(rel.query_id)
        cid = str(rel.doc_id)
        if qid in qmap:
            qrows.append({"query_id": qid, "corpus_id": cid, "question": qmap[qid]})
        if cid not in needed_set and len(needed_ids) < need_docs:
            needed_set.add(cid)
            needed_ids.append(cid)
    docs: list[dict[str, str]] = []
    if not needed_ids:
        return [], qrows
    pending = set(needed_ids)
    for row in ds.docs_iter():
        if str(row.doc_id) not in pending:
            continue
        title = str(getattr(row, "title", "") or "").strip()
        text = str(getattr(row, "text", "") or "").strip()
        if not text:
            pending.discard(str(row.doc_id))
            continue
        docs.append({"doc_id": str(row.doc_id), "title": title, "text": text})
        pending.discard(str(row.doc_id))
        if not pending:
            break
    docs.sort(key=lambda d: needed_ids.index(d["doc_id"]) if d["doc_id"] in needed_ids else 9999)
    return docs, qrows


DATASET_CACHE = REPO / "benchmarks" / "fixtures" / ".dataset_cache"
HOTPOT_JSON_URL = "https://dl.fbaipublicfiles.com/hotpot/hotpot_train_v1.1.json"
HOTPOT_JSON_SHA = "pinned-v1.1"


def _cache_json(name: str, url: str) -> list[dict[str, Any]]:
    DATASET_CACHE.mkdir(parents=True, exist_ok=True)
    path = DATASET_CACHE / name
    if not path.is_file():
        urllib.request.urlretrieve(url, path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_hotpot(_revision: str | None) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = _cache_json(f"hotpot-{HOTPOT_JSON_SHA}.json", HOTPOT_JSON_URL)
    docs: list[dict[str, str]] = []
    questions: list[dict[str, Any]] = []
    for row in rows[:120]:
        ctx = row.get("context") or []
        for title, sentences in ctx:
            doc_id = f"hotpot-{len(docs)}"
            text = " ".join(str(s) for s in sentences)
            docs.append({"doc_id": doc_id, "title": str(title), "text": text})
        answer = str(row.get("answer") or "")
        support = row.get("supporting_facts") or []
        corpus_id = ""
        if support and docs:
            sup_title = str(support[0][0])
            for d in docs[-len(ctx) :]:
                if d["title"] == sup_title:
                    corpus_id = d["doc_id"]
                    break
        if not corpus_id and docs:
            corpus_id = docs[-1]["doc_id"]
        questions.append(
            {
                "query_id": str(row["_id"]),
                "corpus_id": corpus_id,
                "question": str(row["question"]),
                "answer": answer,
            }
        )
    return docs, questions


def _pdf_text(text: str) -> str:
    cleaned = text.replace("\x00", " ").replace("\t", " ")
    return cleaned.encode("latin-1", "replace").decode("latin-1")


def _make_pdf(path: Path, bundle_title: str, passages: Iterable[dict[str, str]]) -> int:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_margins(12, 12, 12)
    pdf.set_auto_page_break(auto=True, margin=15)
    epw = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(epw, 8, _pdf_text(bundle_title))
    pdf.ln(4)
    pages = 1
    for passage in passages:
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(epw, 6, _pdf_text(f"Document ID: {passage['doc_id']}"))
        if passage.get("title"):
            pdf.set_font("Helvetica", "I", 10)
            pdf.multi_cell(epw, 5, _pdf_text(f"Title: {passage['title']}"))
        pdf.set_font("Helvetica", "", 10)
        body = _pdf_text(passage["text"])
        for para in re.split(r"\n{2,}", body):
            chunk = para.strip()
            if not chunk:
                continue
            if pdf.get_y() > 260:
                pdf.add_page()
                pages += 1
            pdf.multi_cell(epw, 5, chunk)
            pdf.ln(2)
        pdf.ln(4)
        if pdf.get_y() > 250:
            pdf.add_page()
            pages += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    return max(1, pages)


def _bundle_docs(
    spec: SourceSpec,
    docs: list[dict[str, str]],
) -> list[tuple[str, list[dict[str, str]]]]:
    need = spec.bundles * spec.docs_per_bundle
    picked = docs[:need]
    bundles: list[tuple[str, list[dict[str, str]]]] = []
    for i in range(spec.bundles):
        chunk = picked[i * spec.docs_per_bundle : (i + 1) * spec.docs_per_bundle]
        if not chunk:
            break
        name = f"ir-{spec.slug}-bundle-{i + 1:02d}.pdf"
        bundles.append((name, chunk))
    return bundles


def _doc_id_to_fixture(doc_ids: set[str], mapping: dict[str, str]) -> list[str]:
    fixtures = {mapping[d] for d in doc_ids if d in mapping}
    return sorted(fixtures)


def _build_questions_for_source(
    spec: SourceSpec,
    doc_to_fixture: dict[str, str],
    docs: list[dict[str, str]],
    qrows: list[dict[str, Any]],
    *,
    existing_ids: set[str],
) -> list[dict[str, Any]]:
    docs_by_id = {d["doc_id"]: d for d in docs}
    cases: list[dict[str, Any]] = []
    tiers_cycle = ["smoke", "core", "core", "advanced", "advanced"]
    idx = 0
    for row in qrows:
        cid = row.get("corpus_id")
        if not cid or cid not in doc_to_fixture:
            continue
        answer_text = str(row.get("answer") or "")
        qtext = str(row["question"])
        gold = docs_by_id.get(cid)
        if gold:
            tokens = _tokens_for_match(f"{gold.get('title', '')} {gold['text'][:600]}", max_tokens=6)
        else:
            tokens = _tokens_for_match(answer_text or qtext)
        if not tokens:
            continue
        tier = tiers_cycle[idx % len(tiers_cycle)]
        idx += 1
        case_id = f"ir-{spec.slug}-{row.get('query_id', idx)}"
        if case_id in existing_ids:
            continue
        existing_ids.add(case_id)
        match: dict[str, Any] = {"any_contains": tokens}
        if answer_text:
            short = answer_text.strip().split()[0] if answer_text.strip() else ""
            if len(short) > 3 and short not in tokens:
                match["any_contains"] = list(dict.fromkeys([short, *tokens]))[:6]
        cases.append(
            {
                "id": case_id,
                "tier": tier,
                "type": "fact" if tier != "advanced" else "overview",
                "suites": ["baseline"],
                "fixture_files": [doc_to_fixture[cid]],
                "expected_pages": [1, 2],
                "question": qtext,
                "answer_match": match,
            }
        )
        if len(cases) >= 12:
            break
    # Multi-doc: pair first two fixtures
    fixtures = sorted(set(doc_to_fixture.values()))
    if len(fixtures) >= 2 and f"ir-{spec.slug}-multi" not in existing_ids:
        existing_ids.add(f"ir-{spec.slug}-multi")
        cases.append(
            {
                "id": f"ir-{spec.slug}-multi",
                "tier": "advanced",
                "type": "multi_doc",
                "suites": ["baseline"],
                "fixture_files": fixtures[:2],
                "expected_pages": [1],
                "question": f"Across the bundled {spec.slug} documents, what topic domain do the titles suggest?",
                "answer_match": {"any_contains": _tokens_for_match(spec.slug + " " + (fixtures[0][:20]), max_tokens=3)},
            }
        )
    if f"ir-{spec.slug}-out-of-scope" not in existing_ids and fixtures:
        existing_ids.add(f"ir-{spec.slug}-out-of-scope")
        cases.append(
            {
                "id": f"ir-{spec.slug}-out-of-scope",
                "tier": "smoke" if spec.slug == "nfcorpus" else "core",
                "type": "out_of_scope",
                "suites": ["baseline", "smoke"] if spec.slug == "nfcorpus" else ["baseline"],
                "fixture_files": [fixtures[0]],
                "expected_pages": [],
                "question": "What is the capital city of Australia?",
                "answer_match": {"decline": True},
            }
        )
    return cases


def _load_source(spec: SourceSpec) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    if spec.skipped:
        return [], []
    if spec.slug in IR_DATASETS_BEIR:
        return _load_beir_ir(spec)
    revision = HF_DATASET_PINS.get(spec.hf_id)
    if spec.hf_id == "hotpot_qa":
        return _load_hotpot(revision)
    raise ValueError(spec.hf_id)


def _merge_manifest(new_papers: list[dict[str, Any]]) -> None:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    by_file = {p["fixture_file"]: p for p in data.get("papers") or []}
    for paper in new_papers:
        by_file[paper["fixture_file"]] = paper
    data["papers"] = sorted(by_file.values(), key=lambda p: p["fixture_file"])
    MANIFEST_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _append_licenses_md(rows: list[str]) -> None:
    text = LICENSES_APPEND_PATH.read_text(encoding="utf-8")
    marker = "## IR corpus PDFs (generated)"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    block = [marker, "", "| File | Source | License | Notes |", "| --- | --- | --- | --- |"]
    block.extend(rows)
    LICENSES_APPEND_PATH.write_text(text + "\n".join(block) + "\n", encoding="utf-8")


def _trim_smoke_tier(questions: list[dict[str, Any]], *, max_smoke: int = 6) -> None:
    smoke = [q for q in questions if q.get("tier") == "smoke"]
    while len(smoke) > max_smoke:
        demoted = False
        for q in questions:
            if q.get("tier") != "smoke" or q.get("type") == "out_of_scope":
                continue
            if q["id"].startswith(("plos", "invasion", "stats", "wolbachia", "ir-")):
                q["tier"] = "core"
                demoted = True
                break
        smoke = [x for x in questions if x.get("tier") == "smoke"]
        if not demoted:
            break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print planned bundles only")
    parser.add_argument("--merge-questions", action="store_true", help="Append generated cases to benchmarks/questions.json")
    args = parser.parse_args()

    new_papers: list[dict[str, Any]] = []
    license_rows: list[str] = []
    all_new_cases: list[dict[str, Any]] = []
    skipped_report: list[str] = []

    for spec in SOURCES:
        if spec.skipped:
            skipped_report.append(f"{spec.slug}: {spec.skip_reason}")
            continue
        docs, qrows = _load_source(spec)
        bundles = _bundle_docs(spec, docs)
        doc_to_fixture: dict[str, str] = {}
        for fname, passages in bundles:
            for p in passages:
                doc_to_fixture[p["doc_id"]] = fname
        if args.dry_run:
            print(f"{spec.slug}: {len(bundles)} PDFs, {len(qrows)} qrels")
            continue
        for fname, passages in bundles:
            out = BUNDLED_DIR / fname
            pages = _make_pdf(out, f"IR eval bundle ({spec.slug})", passages)
            sha = _sha256(out)
            new_papers.append(
                {
                    "id": fname.removesuffix(".pdf"),
                    "fixture_file": fname,
                    "title": f"IR eval bundle: {spec.slug}",
                    "authors": f"askPDF eval corpus ({spec.slug})",
                    "pages": pages,
                    "source_url": IR_DATASETS_BEIR.get(spec.slug, f"https://huggingface.co/datasets/{spec.hf_id}"),
                    "license": spec.license_name,
                    "sha256": sha,
                    "tags": ["ir", spec.slug],
                }
            )
            license_rows.append(
                f"| `{fname}` | {spec.hf_id} | {spec.license_name} | {spec.license_notes} |"
            )

    if args.dry_run:
        for line in skipped_report:
            print("SKIP", line)
        return 0

    if new_papers:
        _merge_manifest(new_papers)
        _append_licenses_md(license_rows)

    if args.merge_questions:
        data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
        questions = list(data.get("questions") or [])
        existing_ids = {q["id"] for q in questions}
        for spec in SOURCES:
            if spec.skipped:
                continue
            docs, qrows = _load_source(spec)
            bundles = _bundle_docs(spec, docs)
            doc_to_fixture = {}
            for fname, passages in bundles:
                for p in passages:
                    doc_to_fixture[p["doc_id"]] = fname
            all_new_cases.extend(
                _build_questions_for_source(spec, doc_to_fixture, docs, qrows, existing_ids=existing_ids)
            )
        # De-duplicate by id
        by_id = {q["id"]: q for q in questions}
        for case in all_new_cases:
            by_id[case["id"]] = case
        merged = list(by_id.values())
        _trim_smoke_tier(merged)
        data["description"] = (
            "Bundled CC BY / CC BY-SA IR and PLOS benchmark corpus (see benchmarks/CORPUS.md)."
        )
        data["questions"] = merged
        QUESTIONS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"merged {len(all_new_cases)} IR cases into {QUESTIONS_PATH}")

    meta_path = REPO / "benchmarks" / "fixtures" / "corpus_build_metadata.json"
    meta = {
        "ir_datasets": IR_DATASETS_BEIR,
        "hf_dataset_pins": HF_DATASET_PINS,
        "skipped": skipped_report,
        "added_pdfs": [p["fixture_file"] for p in new_papers],
    }
    if new_papers:
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())