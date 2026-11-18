from runtime_protocol.permission_modes import derive_browser_access_mode, derive_web_search_mode


def test_derive_web_search_mode_prefers_hitl_over_use_web_search():
    assert derive_web_search_mode(hitl_web_approval=True, use_web_search=True) == "ask"
    assert derive_web_search_mode(use_web_search=True) == "on"
    assert derive_web_search_mode(use_web_search=False) == "off"
    assert derive_web_search_mode(web_search_mode="ask", use_web_search=False) == "ask"


def test_derive_browser_access_mode_matches_web_semantics():
    assert derive_browser_access_mode(hitl_browser_approval=True, use_browser_access=True) == "ask"
    assert derive_browser_access_mode(use_browser_access=True) == "on"