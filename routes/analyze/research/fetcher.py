"""
Content Fetcher - Browser-based content extraction using Playwright.

Handles JS-rendered content, paywalls, modern web pages.
"""

import hashlib
from typing import Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_OK = True
except ImportError:
    PLAYWRIGHT_OK = False


@dataclass
class FetchResult:
    url: str
    title: str
    content: str
    success: bool
    error: Optional[str] = None


def normalize_url(url: str) -> str:
    """Extract real URL from tracking redirects."""
    # Bing tracking
    if 'bing.com/ck/a' in url:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if 'u' in params:
            encoded = params['u'][0]
            if encoded.startswith('a1'):
                import base64
                try:
                    return base64.b64decode(encoded[2:]).decode('utf-8')
                except:
                    pass

    # Google tracking
    if 'google.com/url' in url:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if 'url' in params:
            return params['url'][0]
        if 'q' in params:
            return params['q'][0]

    return url


def make_doc_id(source: str, url: str) -> str:
    return f"{source}_{hashlib.md5(url.encode()).hexdigest()[:12]}"


class BrowserFetcher:
    """
    Fetches web content using a real browser.
    Handles JS, dynamic content, etc.
    """

    def __init__(self, headless: bool = True, timeout: int = 30000):
        self.headless = headless
        self.timeout = timeout
        self._browser = None
        self._playwright = None

    def __enter__(self):
        if not PLAYWRIGHT_OK:
            raise RuntimeError("playwright not installed")
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        return self

    def __exit__(self, *args):
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def fetch(self, url: str) -> FetchResult:
        """Fetch URL and extract content."""
        real_url = normalize_url(url)

        try:
            page = self._browser.new_page()
            page.set_extra_http_headers({
                'Accept-Language': 'en-US,en;q=0.9',
            })

            page.goto(real_url, timeout=self.timeout, wait_until='domcontentloaded')

            # Wait for content to load
            page.wait_for_timeout(2000)

            title = page.title()

            # Extract text content
            content = page.evaluate('''() => {
                // Remove junk
                ['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript', 'iframe'].forEach(tag => {
                    document.querySelectorAll(tag).forEach(el => el.remove());
                });

                // Try to find main content
                const article = document.querySelector('article') ||
                               document.querySelector('main') ||
                               document.querySelector('[class*="article"]') ||
                               document.querySelector('[class*="content"]') ||
                               document.querySelector('[id*="content"]');

                const el = article || document.body;
                return el ? el.innerText : '';
            }''')

            page.close()

            # Clean up
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            content = '\n'.join(lines)

            if len(content) > 50000:
                content = content[:50000] + '\n\n[TRUNCATED]'

            if len(content) < 100:
                return FetchResult(real_url, title, "", False, "no content")

            return FetchResult(real_url, title, content, True)

        except PlaywrightTimeout:
            return FetchResult(real_url, "", "", False, "timeout")
        except Exception as e:
            return FetchResult(real_url, "", "", False, str(e))


def fetch_url(url: str, timeout: int = 30000) -> FetchResult:
    """Convenience function to fetch a single URL."""
    if not PLAYWRIGHT_OK:
        return FetchResult(url, "", "", False, "playwright not installed")

    with BrowserFetcher(timeout=timeout) as fetcher:
        return fetcher.fetch(url)
