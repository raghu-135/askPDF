import logging

from app.agent.agent_helpers import parse_intent_response


def test_parse_intent_response_accepts_first_person_clarification_options():
    raw = """
    <route>CLARIFY</route>
    <rewritten_query>I don't get it.</rewritten_query>
    <reference_type>SEMANTIC</reference_type>
    <context_coverage>PARTIAL</context_coverage>
    <clarification_options>
      <option>What simpler explanation do I need for this concept?</option>
      <option>How do I compare these concepts?</option>
    </clarification_options>
    """

    result = parse_intent_response(raw, logger=logging.getLogger(__name__))

    assert result is not None
    assert result["route"] == "CLARIFY"
    assert result["clarification_options"] == [
        "What simpler explanation do I need for this concept?",
        "How do I compare these concepts?",
    ]


def test_parse_intent_response_rejects_outside_observer_clarification_options():
    raw = """
    <route>CLARIFY</route>
    <rewritten_query>I don't get it.</rewritten_query>
    <reference_type>SEMANTIC</reference_type>
    <context_coverage>PARTIAL</context_coverage>
    <clarification_options>
      <option>Does the user need a simpler explanation of this concept?</option>
      <option>Is the user seeking clarification on how this works?</option>
    </clarification_options>
    """

    result = parse_intent_response(raw, logger=logging.getLogger(__name__))

    assert result is None
