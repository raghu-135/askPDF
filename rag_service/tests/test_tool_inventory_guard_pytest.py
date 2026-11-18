"""Behavior guards for settings catalog vs runtime tool inventory."""

from __future__ import annotations

from app.agent.prompting import build_tool_registry_and_playbook_sections, get_tool_catalog, normalize_tool_instructions
from app.agent.tool_registry import (
    INTERNAL_TOOL_NAMES,
    get_workflow_prompt_tool_names,
    is_accounted_callable_tool_identifier,
    is_internal_tool,
    is_user_configurable_tool,
)
from app.product_orchestration.builtin_workflows import load_builtin_workflows


def test_settings_catalog_includes_browser_and_wikidata():
    catalog = {item["tool_name"]: item for item in get_tool_catalog(for_settings=True)}

    for browser_tool in (
        "browser_navigate",
        "browser_snapshot",
        "browser_act",
        "browser_read",
        "browser_save_to_thread",
    ):
        assert browser_tool in catalog, browser_tool

    assert "wikidata" in catalog
    assert catalog["wikidata"]["id"] == "wikidata_reference"


def test_memory_curator_tools_are_internal_only():
    for tool_name in INTERNAL_TOOL_NAMES:
        assert is_internal_tool(tool_name)
        assert not is_user_configurable_tool(tool_name)


def test_prompt_inventory_tracks_settings_catalog_for_runtime_flags():
    settings_names = {entry["tool_name"] for entry in get_tool_catalog(for_settings=True)}
    for use_web in (True, False):
        for use_browser in (True, False):
            prompt_tools = get_workflow_prompt_tool_names(
                use_web_search=use_web,
                use_browser_access=use_browser,
            )
            assert set(prompt_tools) <= settings_names
            registry_block = build_tool_registry_and_playbook_sections(
                prompt_tools,
                {},
                use_web_search=use_web,
                use_browser_access=use_browser,
            )["TOOL_REGISTRY_SECTION"]
            for tool_name in prompt_tools:
                assert f"`{tool_name}`" in registry_block, (tool_name, use_web, use_browser)


def test_registry_and_playbook_sections_list_the_same_tools():
    tool_names = get_workflow_prompt_tool_names(use_web_search=True, use_browser_access=True)
    sections = build_tool_registry_and_playbook_sections(tool_names, {})
    registry_block = sections["TOOL_REGISTRY_SECTION"]
    playbook_block = sections["TOOL_PLAYBOOK_SECTION"]
    for tool_name in tool_names:
        assert f"`{tool_name}`" in registry_block
        assert f"`{tool_name}`" in playbook_block


def test_builtin_workflow_callable_tools_are_cataloged_or_internal():
    unresolved: set[str] = set()
    for workflow in load_builtin_workflows():
        allowed = (workflow.get("spec_json") or {}).get("config", {}).get("allowed_tool_ids") or []
        for item in allowed:
            item_str = str(item)
            if not is_accounted_callable_tool_identifier(item_str):
                unresolved.add(item_str)
    assert unresolved == set(), sorted(unresolved)


def test_tool_instruction_alias_migrates_legacy_contract_keys():
    normalized = normalize_tool_instructions(
        {
            "search_knowledge": "Search docs",
            "live_web_recon": "Fresh web",
        }
    )
    assert normalized["document_search_knowledge"] == "Search docs"
    assert normalized["live_web_recon"] == "Fresh web"