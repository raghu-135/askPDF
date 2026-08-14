import pytest

from app.tools.context import ToolInvocationContext
from app.tools import provider_clients


class Response:
    def __init__(self, payload=None, *, text="", status=200):
        self._payload = payload
        self.content = text.encode()
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def get(self, *args, **kwargs):
        return next(self.responses)


@pytest.mark.asyncio
async def test_wikipedia_redirect_and_empty_extract_are_normalized(monkeypatch):
    monkeypatch.setattr(provider_clients, "get_http_client", lambda _name: Client([
        Response({"query": {"search": [{"title": "Ada Lovelace"}]}}),
        Response({"query": {"pages": [{"title": "Ada Lovelace", "extract": ""}]}}),
    ]))
    result = await provider_clients.WikipediaProvider().search("Ada", context=ToolInvocationContext())
    assert result.content == "No good Wikipedia Search Result was found"
    assert result.sources == []


@pytest.mark.asyncio
async def test_provider_http_errors_are_isolated(monkeypatch):
    monkeypatch.setattr(provider_clients, "get_http_client", lambda _name: Client([Response({}, status=429)]))
    with pytest.raises(RuntimeError, match="HTTP 429"):
        await provider_clients.WikidataProvider().search("Ada", context=ToolInvocationContext())


@pytest.mark.asyncio
async def test_arxiv_malformed_xml_is_rejected(monkeypatch):
    monkeypatch.setattr(provider_clients, "get_http_client", lambda _name: Client([Response(text="<feed>bad")]))
    with pytest.raises(Exception):
        await provider_clients.ArxivProvider().search("quantum", context=ToolInvocationContext())


@pytest.mark.asyncio
async def test_pubmed_partial_summary_preserves_available_items(monkeypatch):
    client = Client([
        Response({"esearchresult": {"idlist": ["1", "2"]}}),
        Response({"result": {"1": {"title": "Known", "source": "Journal"}}}),
    ])
    monkeypatch.setattr(provider_clients, "get_http_client", lambda _name: client)
    result = await provider_clients.PubMedProvider().search("topic", context=ToolInvocationContext())
    assert "Known" in result.content
    assert "2" in result.content


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [
    provider_clients.SemanticScholarProvider,
    provider_clients.StackExchangeProvider,
    provider_clients.YahooFinanceNewsProvider,
])
async def test_provider_empty_results_have_stable_text(monkeypatch, provider):
    monkeypatch.setattr(provider_clients, "get_http_client", lambda _name: Client([Response({})]))
    result = await provider().search("query", context=ToolInvocationContext())
    assert result.sources == []
    assert result.content.startswith("No ")


@pytest.mark.asyncio
async def test_yahoo_finance_rejects_non_ticker_queries():
    with pytest.raises(ValueError, match="ticker"):
        await provider_clients.YahooFinanceNewsProvider().search("Nvidia stock news", context=ToolInvocationContext())
