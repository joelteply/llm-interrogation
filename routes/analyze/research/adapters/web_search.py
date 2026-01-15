"""
Web Search Adapter - Search + fetch actual content using browser.

Uses Bing for search, Playwright for content fetching.
"""

import os
from typing import Optional
from ..base import ResearchAdapter, ResearchDoc, AdapterConfig
from ..fetcher import BrowserFetcher, normalize_url, make_doc_id, PLAYWRIGHT_OK

try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False


class WebSearchAdapter(ResearchAdapter):
    name = 'web'
    description = 'Web search with full content fetching'

    def _default_config(self) -> AdapterConfig:
        return AdapterConfig(timeout=30000, max_content_size=50000)

    def available(self) -> bool:
        if not REQUESTS_OK:
            print(f"[{self.name}] requests/bs4 not installed")
            return False
        if not PLAYWRIGHT_OK:
            print(f"[{self.name}] playwright not installed")
            return False
        return True

    def search(self, query: str, limit: int = 10) -> list[ResearchDoc]:
        """Search web and fetch actual content."""
        if not self.available():
            return []

        # Get search results
        search_results = self._search_bing(query, limit * 2)  # Get extra in case some fail

        if not search_results:
            print(f"[{self.name}] No search results")
            return []

        print(f"[{self.name}] Got {len(search_results)} search results, fetching content...")

        # Fetch content using browser
        docs = []
        try:
            with BrowserFetcher(headless=True, timeout=self.config.timeout) as fetcher:
                for result in search_results:
                    if len(docs) >= limit:
                        break

                    url = result['url']
                    print(f"[{self.name}] Fetching: {url[:60]}...")

                    fetch_result = fetcher.fetch(url)

                    if fetch_result.success and fetch_result.content:
                        doc_id = make_doc_id(self.name, fetch_result.url)
                        docs.append(ResearchDoc(
                            id=doc_id,
                            source=self.name,
                            title=fetch_result.title or result['title'],
                            url=fetch_result.url,
                            content=fetch_result.content,
                            metadata={'search_engine': 'bing'}
                        ))
                        print(f"[{self.name}] Got {len(fetch_result.content)} chars")
                    else:
                        print(f"[{self.name}] Failed: {fetch_result.error}")

        except Exception as e:
            print(f"[{self.name}] Browser error: {e}")
            import traceback
            traceback.print_exc()

        print(f"[{self.name}] Returning {len(docs)} documents with content")
        return docs

    def _search_bing(self, query: str, limit: int) -> list[dict]:
        """Search Bing, return list of {title, url}."""
        try:
            url = f'https://www.bing.com/search?q={requests.utils.quote(query)}&count={limit}'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }

            r = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')

            results = []
            for item in soup.select('li.b_algo')[:limit]:
                h2 = item.find('h2')
                link = item.find('a')

                if h2 and link:
                    href = link.get('href', '')
                    real_url = normalize_url(href)
                    results.append({
                        'title': h2.text,
                        'url': real_url,
                    })

            print(f"[{self.name}] Bing found {len(results)} results")
            return results

        except Exception as e:
            print(f"[{self.name}] Bing search error: {e}")
            return []
