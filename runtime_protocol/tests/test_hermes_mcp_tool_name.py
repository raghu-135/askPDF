import unittest

from runtime_protocol.hermes_mcp_tool_name import mcp_prefixed_tool_name


class HermesMcpToolNameTests(unittest.TestCase):
    def test_short_name_unchanged(self):
        name = mcp_prefixed_tool_name("askpdf_abcd", "search_documents")
        assert name == "mcp__askpdf_abcd__search_documents"

    def test_long_name_is_clamped_with_hash_suffix(self):
        long_tool = "search_thread_conversation_history"
        server = "askpdf_" + ("a" * 20)
        name = mcp_prefixed_tool_name(server, long_tool)
        assert len(name) == 64
        assert name.startswith("mcp__")
        assert name.endswith("_" + name.split("_")[-1])  # stable 8-hex hash suffix
