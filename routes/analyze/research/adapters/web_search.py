"""
Web Search Adapter - DuckDuckGo with Bing fallback.
"""

import hashlib
from ..base import ResearchAdapter, ResearchDoc

# Check what's available
try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False

try:
    import requests
    from bs4 import BeautifulSoup
    BING_AVAILABLE = True
except ImportError:
    BING_AVAILABLE = False


class WebSearchAdapter(ResearchAdapter):
    name = 'web'
    description = 'Web search via DuckDuckGo/Bing'

    def available(self) -> bool:
        return DDG_AVAILABLE or BING_AVAILABLE

    def search(self, query: str, limit: int = 10) -> list[ResearchDoc]:
        # Try DuckDuckGo first
        results = self._search_ddg(query, limit) if DDG_AVAILABLE else []

        # Fallback to Bing
        if not results and BING_AVAILABLE:
            results = self._search_bing(query, limit)

        return results

    def _search_ddg(self, query: str, limit: int) -> list[ResearchDoc]:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=limit))

            docs = []
            for r in results:
                title = r.get('title', '')
                body = r.get('body', r.get('snippet', ''))
                url = r.get('href', r.get('link', ''))

                if title:
                    doc_id = f"web_{hashlib.md5(url.encode()).hexdigest()[:12]}"
                    docs.append(ResearchDoc(
                        id=doc_id,
                        source=self.name,
                        title=title,
                        url=url,
                        content=f"{title}\n\n{body}",
                        metadata={'search_engine': 'duckduckgo'}
                    ))

            print(f"[{self.name}] DDG found {len(docs)} results")
            return docs

        except Exception as e:
            print(f"[{self.name}] DDG error: {e}")
            return []

    def _search_bing(self, query: str, limit: int) -> list[ResearchDoc]:
        try:
            url = f'https://www.bing.com/search?q={requests.utils.quote(query)}&count={limit}'
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}

            r = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')

            docs = []
            for item in soup.select('li.b_algo')[:limit]:
                h2 = item.find('h2')
                snippet = item.find('p')
                link = item.find('a')

                if h2:
                    title = h2.text
                    body = snippet.text if snippet else ''
                    href = link.get('href', '') if link else ''

                    doc_id = f"web_{hashlib.md5(href.encode()).hexdigest()[:12]}"
                    docs.append(ResearchDoc(
                        id=doc_id,
                        source=self.name,
                        title=title,
                        url=href,
                        content=f"{title}\n\n{body}",
                        metadata={'search_engine': 'bing'}
                    ))

            print(f"[{self.name}] Bing found {len(docs)} results")
            return docs

        except Exception as e:
            print(f"[{self.name}] Bing error: {e}")
            return []
