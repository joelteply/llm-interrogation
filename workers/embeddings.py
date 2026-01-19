"""
Embeddings service - generates semantic embeddings for source content.

Supports multiple providers: OpenAI, Groq, Fireworks, etc.
Uses OpenAI API format which most providers support.
"""

import os
import numpy as np
from typing import Optional
from openai import OpenAI


# Default embedding model - fast and good quality
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_PROVIDER = "openai"

# Embedding dimensions by model
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def get_embedding_client(provider: str = DEFAULT_EMBEDDING_PROVIDER) -> OpenAI:
    """Get OpenAI-compatible client for embeddings."""
    if provider == "openai":
        return OpenAI()
    elif provider == "groq":
        # Groq doesn't support embeddings yet, fall back to OpenAI
        return OpenAI()
    elif provider == "fireworks":
        return OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ.get("FIREWORKS_API_KEY")
        )
    elif provider == "together":
        return OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.environ.get("TOGETHER_API_KEY")
        )
    elif provider == "openrouter":
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY")
        )
    else:
        # Default to OpenAI
        return OpenAI()


def embed_text(
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
    provider: str = DEFAULT_EMBEDDING_PROVIDER
) -> list[float]:
    """Generate embedding for a single text."""
    if not text.strip():
        return []

    client = get_embedding_client(provider)

    # Truncate if too long (most models have 8K token limit)
    max_chars = 30000  # Rough approximation
    if len(text) > max_chars:
        text = text[:max_chars]

    response = client.embeddings.create(
        model=model,
        input=text
    )

    return response.data[0].embedding


def embed_texts(
    texts: list[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    provider: str = DEFAULT_EMBEDDING_PROVIDER,
    batch_size: int = 100
) -> list[list[float]]:
    """Generate embeddings for multiple texts in batches."""
    if not texts:
        return []

    client = get_embedding_client(provider)
    embeddings = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Filter empty strings and truncate long ones
        max_chars = 30000
        processed = []
        indices = []
        for j, text in enumerate(batch):
            if text.strip():
                processed.append(text[:max_chars] if len(text) > max_chars else text)
                indices.append(i + j)

        if not processed:
            # All empty, return zero vectors
            for _ in batch:
                embeddings.append([])
            continue

        response = client.embeddings.create(
            model=model,
            input=processed
        )

        # Map results back, using empty for skipped texts
        batch_embeddings = [[] for _ in batch]
        for k, emb in enumerate(response.data):
            original_idx = indices[k] - i
            batch_embeddings[original_idx] = emb.embedding

        embeddings.extend(batch_embeddings)

    return embeddings


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec1 or not vec2:
        return 0.0

    a = np.array(vec1)
    b = np.array(vec2)

    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


def find_similar(
    query_embedding: list[float],
    candidates: list[tuple[str, list[float]]],  # (id, embedding) pairs
    top_k: int = 10,
    threshold: float = 0.0
) -> list[tuple[str, float]]:
    """Find most similar items to query embedding.

    Returns list of (id, similarity_score) tuples, sorted by score descending.
    """
    if not query_embedding or not candidates:
        return []

    scores = []
    for item_id, embedding in candidates:
        if embedding:
            score = cosine_similarity(query_embedding, embedding)
            if score >= threshold:
                scores.append((item_id, score))

    # Sort by score descending
    scores.sort(key=lambda x: -x[1])

    return scores[:top_k]


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence/paragraph boundary
        if end < len(text):
            # Look for newline first
            newline_pos = text.rfind('\n', start + chunk_size // 2, end)
            if newline_pos > start:
                end = newline_pos + 1
            else:
                # Look for sentence end
                for sep in ['. ', '? ', '! ', '; ']:
                    sep_pos = text.rfind(sep, start + chunk_size // 2, end)
                    if sep_pos > start:
                        end = sep_pos + len(sep)
                        break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def chunk_code(content: str, language: str = "python") -> list[str]:
    """Split code into semantic chunks (functions, classes).

    Returns list of code chunks with context.
    """
    chunks = []
    lines = content.split('\n')

    if language in ("python", "py"):
        # Find class/function definitions
        current_chunk = []
        current_indent = 0
        in_block = False
        block_start_line = 0

        for i, line in enumerate(lines):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            # Check for new definition
            if stripped.startswith(('def ', 'class ', 'async def ')):
                # Save previous chunk if exists
                if current_chunk and in_block:
                    chunks.append('\n'.join(current_chunk))

                current_chunk = [line]
                current_indent = indent
                in_block = True
                block_start_line = i

            elif in_block:
                # Check if we're still in the block
                if stripped and indent <= current_indent and i > block_start_line:
                    # New block at same or lower indent - save and start new
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    in_block = stripped.startswith(('def ', 'class ', 'async def '))
                    if in_block:
                        current_indent = indent
                        block_start_line = i
                else:
                    current_chunk.append(line)
            else:
                # Not in a block, check for module-level code
                if stripped and not stripped.startswith('#'):
                    current_chunk.append(line)

        # Don't forget last chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

    elif language in ("javascript", "typescript", "js", "ts"):
        # Find function/class definitions by braces
        current_chunk = []
        brace_depth = 0
        in_definition = False

        for line in lines:
            stripped = line.strip()

            # Check for function/class start
            if any(kw in stripped for kw in ['function ', 'class ', 'const ', 'let ', 'export ']):
                if '{' in stripped:
                    in_definition = True

            if in_definition or brace_depth > 0:
                current_chunk.append(line)

            # Track braces
            brace_depth += stripped.count('{') - stripped.count('}')

            if brace_depth == 0 and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                in_definition = False

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

    else:
        # Generic: just use fixed-size chunks
        chunks = chunk_text(content, chunk_size=1500, overlap=200)

    return [c for c in chunks if c.strip()]
