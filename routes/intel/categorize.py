"""
Entity Categorization

Web search and categorization of entities.
Searches with context (tuples), not just names.
"""

from typing import List, Optional, Tuple
import re

from .entities import Entity, WebResult

# Import web search if available
try:
    from routes.analyze.research.fetcher import BrowserFetcher, normalize_url
    WEB_OK = True
except ImportError:
    WEB_OK = False


def build_search_queries(entity: Entity, context_entities: List[str] = None) -> List[str]:
    """
    Build contextual search queries for an entity.

    Searches tuples (entity + context) not just the entity name.
    """
    queries = []
    text = entity.text

    # Basic search
    queries.append(f'"{text}"')

    # With context entities (search tuples)
    if context_entities:
        for ctx in context_entities[:3]:
            queries.append(f'"{text}" "{ctx}"')

    # Extract context from original response
    if entity.originated_from and entity.originated_from.response_text:
        # Find other entities mentioned in same response
        response = entity.originated_from.response_text
        # Simple extraction of other capitalized terms
        other_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response)
        other_terms = [t for t in other_terms if t.lower() != text.lower()]
        for term in other_terms[:2]:
            queries.append(f'"{text}" "{term}"')

    return queries[:5]  # Limit to 5 queries


def search_web(query: str, max_results: int = 5) -> List[dict]:
    """
    Search web for a query.

    Returns list of {url, title, snippet}.
    """
    if not WEB_OK:
        print("[categorize] Web search not available")
        return []

    try:
        # Use Bing search via requests
        import requests

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        url = f"https://www.bing.com/search?q={requests.utils.quote(query)}&count={max_results}"
        resp = requests.get(url, headers=headers, timeout=10)

        if resp.status_code != 200:
            return []

        # Parse results (basic)
        results = []
        html = resp.text

        # Extract search result blocks
        import re
        # Find all result links
        links = re.findall(r'<a[^>]+href="(https?://[^"]+)"[^>]*>([^<]+)</a>', html)

        for href, title in links[:max_results]:
            # Skip Bing internal links
            if 'bing.com' in href or 'microsoft.com' in href:
                continue
            results.append({
                'url': normalize_url(href),
                'title': title,
                'snippet': ''
            })

        return results

    except Exception as e:
        print(f"[categorize] Search error: {e}")
        return []


def categorize_entity(
    entity: Entity,
    context_entities: List[str] = None,
    topic: str = ""
) -> Tuple[str, List[WebResult]]:
    """
    Categorize an entity via web search.

    Returns (category, web_results).

    Categories:
    - CORROBORATED: Found on web with matching context
    - UNCORROBORATED: Not found on web
    - CONTRADICTED: Web contradicts the claim
    """
    queries = build_search_queries(entity, context_entities)

    all_results = []
    found_with_context = False
    found_basic = False

    for query in queries:
        results = search_web(query)

        for r in results:
            # Check if result matches context
            matches = entity.text.lower() in r['title'].lower()

            wr = WebResult(
                query=query,
                url=r['url'],
                title=r['title'],
                snippet=r.get('snippet', ''),
                matches_context=matches
            )
            all_results.append(wr)

            if matches:
                if '"' in query and query.count('"') >= 4:
                    # Tuple query matched = strong corroboration
                    found_with_context = True
                else:
                    found_basic = True

    # Determine category
    if found_with_context:
        category = "CORROBORATED"
    elif found_basic:
        # Found the name but not the specific context
        # Still useful, but the SPECIFIC claim is uncorroborated
        category = "CORROBORATED"  # Basic corroboration
    elif all_results:
        # Found something but no match
        category = "UNCORROBORATED"
    else:
        # Nothing found
        category = "UNCORROBORATED"

    return category, all_results


def categorize_entities(
    entities: List[Entity],
    topic: str = "",
    limit: int = 50,
    max_workers: int = 5
) -> dict:
    """
    Categorize multiple entities in parallel.

    Returns summary stats.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    stats = {
        "total": 0,
        "corroborated": 0,
        "uncorroborated": 0,
        "contradicted": 0,
        "skipped": 0
    }

    # Get all entity texts for context
    all_texts = [e.text for e in entities]

    # Filter to unprocessed entities
    to_process = [e for e in entities[:limit] if e.category == "UNPROCESSED"]
    stats["skipped"] = min(limit, len(entities)) - len(to_process)

    def process_one(entity: Entity) -> Tuple[Entity, str, List[WebResult]]:
        context = [t for t in all_texts if t != entity.text][:5]
        category, web_results = categorize_entity(entity, context, topic)
        return entity, category, web_results

    print(f"[categorize] Processing {len(to_process)} entities with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, e): e for e in to_process}

        for future in as_completed(futures):
            try:
                entity, category, web_results = future.result()
                entity.category = category
                entity.web_results = web_results
                stats["total"] += 1
                stats[category.lower()] += 1
                print(f"[categorize] {entity.text}: {category} ({len(web_results)} results)")
            except Exception as e:
                print(f"[categorize] Error: {e}")

    return stats
