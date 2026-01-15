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

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    from PyPDF2 import PdfReader
    from io import BytesIO
    PDF_OK = True
except ImportError:
    PDF_OK = False


@dataclass
class FetchResult:
    url: str
    title: str
    content: str
    success: bool
    error: Optional[str] = None


def normalize_url(url: str) -> str:
    """Extract real URL from tracking redirects."""
    import base64
    import re

    # Bing tracking - extract u= parameter directly with regex (query parsing breaks on !&&)
    if 'bing.com/ck/a' in url:
        match = re.search(r'[&?]u=(a1[A-Za-z0-9+/=]+)', url)
        if match:
            encoded = match.group(1)[2:]  # Strip 'a1' prefix
            # Add padding if needed
            padding = 4 - (len(encoded) % 4)
            if padding != 4:
                encoded += '=' * padding
            try:
                decoded = base64.b64decode(encoded).decode('utf-8')
                print(f"[url] Decoded Bing redirect: {decoded[:60]}")
                return decoded
            except Exception as e:
                print(f"[url] Failed to decode Bing URL: {e}")

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


def fetch_pdf(url: str, timeout: int = 30) -> FetchResult:
    """Fetch PDF and extract text."""
    if not REQUESTS_OK:
        return FetchResult(url, "", "", False, "requests not installed")
    if not PDF_OK:
        return FetchResult(url, "", "", False, "PyPDF2 not installed")

    real_url = normalize_url(url)

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/pdf,*/*',
        }
        r = requests.get(real_url, headers=headers, timeout=timeout)
        r.raise_for_status()

        # Check if it's actually a PDF
        content_type = r.headers.get('Content-Type', '')
        if 'pdf' not in content_type.lower() and not real_url.endswith('.pdf'):
            return FetchResult(real_url, "", "", False, f"Not a PDF: {content_type}")

        # Extract text from PDF
        pdf_reader = PdfReader(BytesIO(r.content))
        text_parts = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

        content = '\n\n'.join(text_parts)

        if len(content) < 100:
            return FetchResult(real_url, "", "", False, "PDF has no extractable text")

        # Get title from metadata or URL
        title = ""
        if pdf_reader.metadata and pdf_reader.metadata.title:
            title = pdf_reader.metadata.title
        else:
            title = real_url.split('/')[-1].replace('.pdf', '').replace('-', ' ')

        print(f"[pdf] Extracted {len(content)} chars from {len(pdf_reader.pages)} pages")
        return FetchResult(real_url, title, content, True)

    except Exception as e:
        return FetchResult(real_url, "", "", False, str(e))


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

        # Handle PDFs directly (don't use browser)
        if real_url.endswith('.pdf') or '/pdf/' in real_url:
            return fetch_pdf(real_url)

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
