"""Behavior guards for settings catalog vs runtime tool inventory."""

from __future__ import annotations

from app.agent.prompting import build_tool_registry_and_playbook_sections, get_tool_catalog, normalize_tool_instructions
from app.agent.tool_registry import (
    INTERNAL_TOOL_NAMES,
    get_workflow_prompt_tool_names,
    is_internal_tool,
    is_user_configurable_tool,
    tool_names_for_contract_id,
)
from app.product_orchestration.builtin_workflows import load_builtin_workflows

_WORKFLOW_TOOL_NAME_ALIASES = {
    "pub_med": "pubmed",
    "semanticscholar": "semantic_scholar",
}


def _catalog_tool_names() -> set[str]:
    return {entry["tool_name"] for entry in get_tool_catalog(for_settings=True)}


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
    settings_names = _catalog_tool_names()
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
    catalog_names = _catalog_tool_names()
    unresolved: set[str] = set()
    for workflow in load_builtin_workflows():
        runtime_kind = str((workflow.get("spec_json") or {}).get("runtime", {}).get("kind", ""))
        allowed = (workflow.get("spec_json") or {}).get("config", {}).get("allowed_tool_ids") or []
        for item in allowed:
            item_str = _WORKFLOW_TOOL_NAME_ALIASES.get(str(item), str(item))
            if item_str in catalog_names:
                continue
            resolved = tool_names_for_contract_id(item_str)
            if resolved:
                for tool_name in resolved:
                    if tool_name not in catalog_names and not is_internal_tool(tool_name):
                        unresolved.add(tool_name)
                continue
            if item_str not in catalog_names and not is_internal_tool(item_str):
                if runtime_kind == "hermes_agent" and item_str in catalog_names:
                    continue
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