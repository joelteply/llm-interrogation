"""
Research Module - Fetch context from multiple sources using adapters.

Usage:
    from routes.analyze.research import research
    result = research('query', project_name='my-project')
"""

import json
import random
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

from .base import ResearchAdapter, ResearchDoc
from .cleaner import clean_with_llm, looks_like_garbage
from .adapters import ALL_ADAPTERS


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ResearchResult:
    """Result of a research query."""
    query: str
    sources_used: list[str]
    documents: list[ResearchDoc]
    raw_content: str  # Combined content for LLM
    cached_count: int
    fetched_count: int


# =============================================================================
# Cache Management
# =============================================================================

PROJECTS_DIR = Path(__file__).parent.parent.parent.parent / "projects"


def get_cache_dir(project_name: Optional[str] = None) -> Path:
    """Get research directory within project."""
    if project_name:
        cache_dir = PROJECTS_DIR / project_name / "research"
    else:
        cache_dir = PROJECTS_DIR / "_global" / "research"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def cache_document(doc: ResearchDoc, project_name: Optional[str] = None) -> Optional[str]:
    """Save document to cache. Returns doc ID or None if discarded."""
    cache_dir = get_cache_dir(project_name)

    content = doc.content

    # Clean garbage OCR
    if looks_like_garbage(content):
        content = clean_with_llm(content, doc.title)
        if content is None:
            return None

    # Save as .md
    md_path = cache_dir / f"{doc.id}.md"
    with open(md_path, 'w') as f:
        f.write(f"# {doc.title}\n\n")
        f.write(f"**Source:** {doc.source}  \n")
        f.write(f"**URL:** {doc.url}  \n")
        f.write(f"**Fetched:** {datetime.now(timezone.utc).isoformat()}\n\n")
        f.write("---\n\n")
        f.write(content)

    # Update index
    _update_index(cache_dir, doc)
    print(f"[RESEARCH] Cached {doc.id}: {doc.title[:40]}...")
    return doc.id


def get_cached_document(doc_id: str, project_name: Optional[str] = None) -> Optional[ResearchDoc]:
    """Retrieve document from cache."""
    cache_dir = get_cache_dir(project_name)
    md_path = cache_dir / f"{doc_id}.md"

    if not md_path.exists():
        return None

    with open(md_path) as f:
        content = f.read()

    # Get metadata from index
    index = _load_index(cache_dir)
    doc_info = index.get('documents', {}).get(doc_id, {})

    return ResearchDoc(
        id=doc_id,
        source=doc_info.get('source', 'cache'),
        title=doc_info.get('title', doc_id),
        url=doc_info.get('url', ''),
        content=content,
    )


def list_cached_documents(project_name: Optional[str] = None) -> list[dict]:
    """List cached documents (metadata only)."""
    cache_dir = get_cache_dir(project_name)
    index = _load_index(cache_dir)

    return [
        {
            'id': doc_id,
            'title': info.get('title', ''),
            'source': info.get('source', ''),
            'fetched_at': info.get('fetched_at', ''),
        }
        for doc_id, info in index.get('documents', {}).items()
    ]


def _load_index(cache_dir: Path) -> dict:
    index_path = cache_dir / "_index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)
    return {'documents': {}}


def _update_index(cache_dir: Path, doc: ResearchDoc):
    index_path = cache_dir / "_index.json"
    index = _load_index(cache_dir)

    index['documents'][doc.id] = {
        'title': doc.title,
        'source': doc.source,
        'url': doc.url,
        'fetched_at': datetime.now(timezone.utc).isoformat(),
    }

    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)


# =============================================================================
# Adapter Management
# =============================================================================

def get_adapters(sources: list[str] = None) -> list[ResearchAdapter]:
    """Get available adapters, optionally filtered by source names."""
    adapters = [cls() for cls in ALL_ADAPTERS]
    available = [a for a in adapters if a.available()]

    if sources:
        available = [a for a in available if a.name in sources]

    return available


def list_sources() -> list[dict]:
    """List all registered sources and their availability."""
    return [
        {
            'name': cls.name,
            'description': cls.description,
            'available': cls().available(),
        }
        for cls in ALL_ADAPTERS
    ]


# =============================================================================
# Main Research Function
# =============================================================================

def research(
    query: str,
    project_name: Optional[str] = None,
    sources: list[str] = None,
    max_per_source: int = 10,
    include_cache: bool = True
) -> ResearchResult:
    """
    Research a query across multiple sources.

    Args:
        query: What to search for
        project_name: Project to cache results in
        sources: Which sources to use (None = all available)
        max_per_source: Max results per source
        include_cache: Include previously cached docs

    Returns:
        ResearchResult with raw content for LLM
    """
    adapters = get_adapters(sources)
    random.shuffle(adapters)

    all_docs = []
    raw_chunks = []
    sources_used = []
    cached_count = 0
    fetched_count = 0

    # Check cache first
    if include_cache and project_name:
        cached = list_cached_documents(project_name)
        if cached:
            sources_used.append('cache')
            random.shuffle(cached)
            for doc_info in cached[:max_per_source]:
                doc = get_cached_document(doc_info['id'], project_name)
                if doc and doc.content:
                    all_docs.append(doc)
                    raw_chunks.append(f"[Cached: {doc.title}]\n{doc.content[:3000]}")
                    cached_count += 1

    # Query each adapter
    for adapter in adapters:
        try:
            docs = adapter.search(query, limit=max_per_source)
            if docs:
                sources_used.append(adapter.name)

            for doc in docs:
                # Check if already cached
                existing = get_cached_document(doc.id, project_name)
                if existing:
                    all_docs.append(existing)
                    raw_chunks.append(f"[{adapter.name}: {existing.title}]\n{existing.content[:3000]}")
                    cached_count += 1
                else:
                    # Cache new doc
                    if cache_document(doc, project_name):
                        all_docs.append(doc)
                        raw_chunks.append(f"[{adapter.name}: {doc.title}]\n{doc.content[:3000]}")
                        fetched_count += 1

        except Exception as e:
            print(f"[RESEARCH] {adapter.name} error: {e}")

    # Combine raw content
    random.shuffle(raw_chunks)
    raw_content = '\n\n---\n\n'.join(raw_chunks[:15])

    return ResearchResult(
        query=query,
        sources_used=list(set(sources_used)),
        documents=all_docs,
        raw_content=raw_content,
        cached_count=cached_count,
        fetched_count=fetched_count
    )


# =============================================================================
# Convenience / Backwards Compatibility
# =============================================================================

def research_topic(topic: str, max_results: int = 8) -> str:
    """Quick web search. Returns formatted text."""
    adapters = get_adapters(['web'])
    if not adapters:
        return "Web search not available"

    docs = adapters[0].search(topic, limit=max_results)
    if not docs:
        return "No results found"

    lines = ["Web search results:"]
    for doc in docs:
        lines.append(f"  - {doc.title}: {doc.content[:150]}...")

    return "\n".join(lines)
