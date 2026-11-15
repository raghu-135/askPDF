from runtime_protocol.research_gaps import (
    filter_answer_format_meta_limitations,
    is_answer_format_meta_limitation,
)


def test_filters_one_sentence_and_title_only_meta_issues():
    issues = [
        "The answer should be one sentence only.",
        "The factual claim is unsupported.",
    ]
    filtered = filter_answer_format_meta_limitations(issues)
    assert filtered == ["The factual claim is unsupported."]
    assert is_answer_format_meta_limitation("h1 text was not captured in evidence")