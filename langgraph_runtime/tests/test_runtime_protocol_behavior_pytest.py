import pytest

from runtime_protocol.contracts import RuntimeBehaviorDescriptor


def _behavior():
    return {
        "continuation_semantics": "same_run_safe_boundary",
        "usage_accounting_owner": "runtime",
        "preserves_run_id": True,
        "artifact_inheritance": "valid_artifacts",
        "supports_orchestration_delta": True,
        "required_input_fields": ["task_context"],
        "supports_pause_resume": True,
        "supports_course_correction": True,
        "budget_boundary_owner": "runtime",
        "grounding_owner": "runtime",
    }


def test_behavior_descriptor_round_trips_canonically():
    descriptor = RuntimeBehaviorDescriptor.from_mapping(_behavior())
    assert descriptor.to_dict() == _behavior()


def test_behavior_descriptor_rejects_boolean_coercion_and_unknown_values():
    with pytest.raises(TypeError):
        RuntimeBehaviorDescriptor.from_mapping({**_behavior(), "supports_course_correction": "false"})
    with pytest.raises(ValueError):
        RuntimeBehaviorDescriptor.from_mapping({**_behavior(), "continuation_semantics": "same_run"})
    with pytest.raises(ValueError):
        RuntimeBehaviorDescriptor.from_mapping({**_behavior(), "budget_boundary_owner": "runtimee"})


def test_behavior_descriptor_rejects_incomplete_and_extra_metadata():
    with pytest.raises(ValueError):
        RuntimeBehaviorDescriptor.from_mapping({key: value for key, value in _behavior().items() if key != "grounding_owner"})
    with pytest.raises(ValueError):
        RuntimeBehaviorDescriptor.from_mapping({**_behavior(), "legacy_fallback": True})
