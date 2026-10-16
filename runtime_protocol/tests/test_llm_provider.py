import unittest

from runtime_protocol.llm_provider import (
    LLM_SDK_PLACEHOLDER_API_KEY,
    LlmProviderConfigurationError,
    llm_provider_configuration,
    normalize_llm_api_url,
    openai_sdk_default_headers,
)


class LlmProviderTests(unittest.TestCase):
    def test_normalize_appends_v1_once(self):
        self.assertEqual(normalize_llm_api_url("http://host:11434"), "http://host:11434/v1")
        self.assertEqual(normalize_llm_api_url("http://host:1234/v1/"), "http://host:1234/v1")

    def test_missing_url_fails_closed(self):
        with self.assertRaises(LlmProviderConfigurationError):
            llm_provider_configuration(environ={"OPENAI_API_KEY": "sk-test"})

    def test_key_sends_bearer_and_empty_key_sends_no_authorization(self):
        hosted = llm_provider_configuration(
            environ={"LLM_API_URL": "https://openrouter.ai/api/v1", "OPENAI_API_KEY": "sk-or-test"},
        )
        self.assertEqual(hosted.request_headers, {"Authorization": "Bearer sk-or-test"})
        self.assertEqual(hosted.sdk_api_key, "sk-or-test")
        local = llm_provider_configuration(
            environ={"LLM_API_URL": "http://host.docker.internal:1234/v1", "OPENAI_API_KEY": ""},
        )
        self.assertEqual(local.request_headers, {})
        self.assertEqual(local.sdk_api_key, LLM_SDK_PLACEHOLDER_API_KEY)
        self.assertEqual(openai_sdk_default_headers(hosted.request_headers), None)
