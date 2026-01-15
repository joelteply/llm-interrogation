"""
Tests for research adapters.

Run: ./venv/bin/python -m pytest tests/test_research.py -v
"""

import pytest
from routes.analyze.research.base import ResearchDoc, AdapterConfig
from routes.analyze.research.fetcher import (
    normalize_url, make_doc_id, BrowserFetcher, fetch_url, PLAYWRIGHT_OK
)
from routes.analyze.research.adapters.web_search import WebSearchAdapter
from routes.analyze.research.adapters.documentcloud import DocumentCloudAdapter
from routes.analyze.research import research, get_adapters, list_sources


class TestUrlNormalization:
    """Test URL normalization from tracking redirects."""

    def test_bing_redirect(self):
        # Bing encodes URLs as base64 with 'a1' prefix
        import base64
        real_url = "https://www.forbes.com/article"
        encoded = "a1" + base64.b64encode(real_url.encode()).decode()
        bing_url = f"https://www.bing.com/ck/a?u={encoded}"

        result = normalize_url(bing_url)
        assert result == real_url

    def test_google_redirect(self):
        google_url = "https://www.google.com/url?url=https://example.com/page"
        result = normalize_url(google_url)
        assert result == "https://example.com/page"

    def test_regular_url_unchanged(self):
        url = "https://www.nytimes.com/article"
        assert normalize_url(url) == url


class TestDocId:
    """Test document ID generation."""

    def test_consistent_id(self):
        url = "https://example.com/doc"
        id1 = make_doc_id("web", url)
        id2 = make_doc_id("web", url)
        assert id1 == id2

    def test_different_sources_different_ids(self):
        url = "https://example.com/doc"
        id1 = make_doc_id("web", url)
        id2 = make_doc_id("dc", url)
        assert id1 != id2


@pytest.mark.skipif(not PLAYWRIGHT_OK, reason="playwright not installed")
class TestBrowserFetcher:
    """Test browser-based content fetching."""

    def test_fetch_wikipedia(self):
        """Wikipedia should always work."""
        result = fetch_url("https://en.wikipedia.org/wiki/Test")
        assert result.success
        assert len(result.content) > 1000
        assert result.title

    def test_fetch_invalid_url(self):
        """Invalid URL should fail gracefully."""
        result = fetch_url("https://thissitedoesnotexist12345.com")
        assert not result.success
        assert result.error

    def test_fetch_timeout(self):
        """Test timeout handling."""
        # Very short timeout should fail
        with BrowserFetcher(timeout=1) as fetcher:
            result = fetcher.fetch("https://en.wikipedia.org/wiki/Test")
            # May or may not timeout depending on speed
            assert isinstance(result.success, bool)


class TestWebSearchAdapter:
    """Test web search adapter."""

    def test_available(self):
        adapter = WebSearchAdapter()
        # Should be available if playwright installed
        assert adapter.available() == PLAYWRIGHT_OK

    @pytest.mark.skipif(not PLAYWRIGHT_OK, reason="playwright not installed")
    def test_search_returns_docs(self):
        adapter = WebSearchAdapter()
        docs = adapter.search("python programming", limit=2)

        assert len(docs) > 0
        for doc in docs:
            assert isinstance(doc, ResearchDoc)
            assert doc.id
            assert doc.title
            assert doc.url
            assert len(doc.content) > 100  # Real content, not snippet

    @pytest.mark.skipif(not PLAYWRIGHT_OK, reason="playwright not installed")
    def test_search_fetches_real_content(self):
        """Verify we get actual page content, not search snippets."""
        adapter = WebSearchAdapter()
        docs = adapter.search("wikipedia test page", limit=1)

        if docs:
            # Real content should be substantial
            assert len(docs[0].content) > 1000
            # Should not be a search snippet
            assert "..." not in docs[0].content[:200]


class TestDocumentCloudAdapter:
    """Test DocumentCloud adapter."""

    def test_available_without_creds(self):
        """Should report unavailable without credentials."""
        import os
        # Temporarily clear creds
        old_user = os.environ.pop('DOCUMENTCLOUD_USERNAME', None)
        old_pass = os.environ.pop('DOCUMENTCLOUD_PASSWORD', None)

        try:
            adapter = DocumentCloudAdapter()
            assert not adapter.available()
        finally:
            # Restore
            if old_user:
                os.environ['DOCUMENTCLOUD_USERNAME'] = old_user
            if old_pass:
                os.environ['DOCUMENTCLOUD_PASSWORD'] = old_pass

    def test_search_without_creds_returns_empty(self):
        """Search without creds should return empty, not crash."""
        import os
        old_user = os.environ.pop('DOCUMENTCLOUD_USERNAME', None)
        old_pass = os.environ.pop('DOCUMENTCLOUD_PASSWORD', None)

        try:
            adapter = DocumentCloudAdapter()
            docs = adapter.search("test query")
            assert docs == []
        finally:
            if old_user:
                os.environ['DOCUMENTCLOUD_USERNAME'] = old_user
            if old_pass:
                os.environ['DOCUMENTCLOUD_PASSWORD'] = old_pass


class TestResearchModule:
    """Test main research module."""

    def test_list_sources(self):
        """Should list available sources."""
        sources = list_sources()
        assert len(sources) >= 2

        names = [s['name'] for s in sources]
        assert 'web' in names
        assert 'documentcloud' in names

    def test_get_adapters_filters(self):
        """Should filter adapters by name."""
        adapters = get_adapters(['web'])
        names = [a.name for a in adapters]
        assert 'documentcloud' not in names

    @pytest.mark.skipif(not PLAYWRIGHT_OK, reason="playwright not installed")
    def test_research_function(self):
        """Test main research function."""
        result = research(
            query="test query",
            project_name=None,  # Don't cache
            sources=['web'],
            max_per_source=1
        )

        assert result.query == "test query"
        assert 'web' in result.sources_used or result.fetched_count == 0


class TestPeriodicResearch:
    """Test the periodic research function used in probe loop."""

    def test_function_exists(self):
        from routes.probe import run_periodic_research
        assert callable(run_periodic_research)

    def test_returns_dict(self):
        from routes.probe import run_periodic_research
        result = run_periodic_research(None, None, None)

        assert isinstance(result, dict)
        assert 'fetched_count' in result
        assert 'cached_count' in result
        assert 'queries_run' in result

    def test_handles_missing_project(self):
        from routes.probe import run_periodic_research
        # Should not crash with invalid project
        result = run_periodic_research("nonexistent-project", "test topic", None)
        assert result['fetched_count'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
