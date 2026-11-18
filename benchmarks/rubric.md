# RAG benchmark judge rubric (v1)

Used by `scripts/rag_eval_suite.py` / promptfoo when exact or numeric matching is insufficient.

## Answer correctness (LLM judge)

**Model:** `ASKPDF_BENCHMARK_JUDGE_MODEL` (default `google/gemini-2.5-flash`).

**Inputs:** question, model answer, `answer_match` constraints from the case JSON.

**Score 1 (pass)** if all of the following hold:

- For `decline: true` cases: the answer clearly states the documents do not contain
  the requested information (or equivalent product decline wording), and does not
  invent specific facts.
- For `any_contains`: at least one listed substring is present (case-insensitive),
  OR the judge confirms the answer is factually equivalent to requiring that concept.
- For `all_contains`: every listed substring is present or equivalently stated.
- For `numeric`: the primary numeric claim is within tolerance of `target`.
- The answer does not contradict the reference constraints.

**Score 0 (fail)** otherwise.

Judge prompt is fixed in `scripts/benchmark_scoring.py` (`JUDGE_SYSTEM_PROMPT`); changes require
a benchmark schema bump.

## Retrieval hit

Pass if any retrieved chunk/page in the scored run (child run for Auto) intersects
`expected_pages` for at least one expected file hash, OR `expected_pages` is empty
and the case is `out_of_scope`.

## Latency and cost

- **Latency:** wall-clock from chat POST to terminal run status (seconds).
- **Cost:** sum of `metrics.llm_token_count_total` from the scored run trace when
  present (token proxy, not billed dollars).