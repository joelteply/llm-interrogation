"""
Test fetching actual Epstein documents.

Run: ./venv/bin/python tests/test_epstein_docs.py
"""

from routes.analyze.research import research
from routes.analyze.research.fetcher import fetch_url, BrowserFetcher


def test_fetch_epstein_court_docs():
    """Fetch Epstein court documents via web search."""
    print("\n=== Testing Epstein Court Document Fetch ===\n")

    result = research(
        query='epstein court documents unsealed files',
        project_name=None,  # Don't cache for test
        sources=['web'],
        max_per_source=5
    )

    print(f"Sources used: {result.sources_used}")
    print(f"Fetched: {result.fetched_count}")
    print(f"Documents: {len(result.documents)}")

    assert len(result.documents) > 0, "Should fetch at least one document"

    for doc in result.documents:
        print(f"\n--- {doc.title[:70]} ---")
        print(f"URL: {doc.url}")
        print(f"Content length: {len(doc.content)} chars")
        assert len(doc.content) > 500, f"Document should have real content, got {len(doc.content)}"
        print(f"Preview: {doc.content[:200]}...")

    print("\n✓ PASSED: Fetched real Epstein documents")


def test_fetch_specific_urls():
    """Test fetching specific known Epstein document URLs."""
    print("\n=== Testing Specific URL Fetch ===\n")

    urls = [
        "https://www.courtlistener.com/docket/4355835/giuffre-v-maxwell/",
        "https://www.documentcloud.org/documents/6250471-Epstein-Docs",
        "https://www.justice.gov/usao-sdny/pr/jeffrey-epstein-charged-manhattan-federal-court-sex-trafficking-minors",
    ]

    with BrowserFetcher(headless=True, timeout=30000) as fetcher:
        for url in urls:
            print(f"Fetching: {url[:60]}...")
            result = fetcher.fetch(url)

            if result.success:
                print(f"  ✓ Got {len(result.content)} chars")
                print(f"  Title: {result.title[:60]}")
                assert len(result.content) > 200
            else:
                print(f"  ✗ Failed: {result.error}")

    print("\n✓ PASSED: URL fetch test complete")


def test_fetch_cbs_epstein_files():
    """Test fetching the CBS Epstein files page (JS-heavy)."""
    print("\n=== Testing CBS Epstein Files (JS-rendered) ===\n")

    url = "https://www.cbsnews.com/news/epstein-files-what-we-know/"

    result = fetch_url(url, timeout=30000)

    print(f"URL: {url}")
    print(f"Success: {result.success}")
    print(f"Title: {result.title}")
    print(f"Content length: {len(result.content)} chars")

    if result.success:
        print(f"Preview: {result.content[:300]}...")
        assert len(result.content) > 1000, "CBS article should have substantial content"
        print("\n✓ PASSED: Fetched CBS article with browser")
    else:
        print(f"Error: {result.error}")
        print("✗ FAILED: Could not fetch CBS article")


def test_periodic_research_function():
    """Test the periodic research function directly."""
    print("\n=== Testing Periodic Research Function ===\n")

    from routes.probe import run_periodic_research

    # Test with real project
    result = run_periodic_research(
        project_name="epstein-linked-to-prominent-emails-or-pe",
        topic="Epstein emails gmax1@ellmax.com jeevacation@gmail.com",
        findings=None
    )

    print(f"Fetched: {result['fetched_count']}")
    print(f"Cached: {result['cached_count']}")
    print(f"Queries run: {result['queries_run']}")
    print(f"Error: {result['error']}")

    print("\n✓ PASSED: Periodic research function works")


if __name__ == '__main__':
    test_fetch_epstein_court_docs()
    test_fetch_specific_urls()
    test_fetch_cbs_epstein_files()
    test_periodic_research_function()

    print("\n" + "="*50)
    print("ALL TESTS PASSED")
    print("="*50)
