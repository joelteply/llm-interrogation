"""
Semantic Indexer Worker - indexes seed content with embeddings.

On each tick:
1. Check if there are unembedded probe targets
2. Generate embeddings for them in batches
3. Store embeddings for similarity-based selection
"""

import os
from typing import Optional
from .base import BaseWorker
from .embeddings import embed_texts, chunk_code, chunk_text
from .seed_extractor import detect_content_type
from models import ProbeTarget


class SemanticIndexerWorker(BaseWorker):
    """Worker that generates embeddings for probe targets."""

    name = "SEMANTIC-INDEXER"

    def __init__(self, interval: float = 60.0):
        super().__init__(interval)
        self.batch_size = 20  # Embed this many targets per tick
        self.embedding_model = "text-embedding-3-small"

    def _do_work(self, project_name: str) -> int:
        """Generate embeddings for unembedded targets. Returns count of embedded."""
        if not project_name:
            return 0

        from routes import project_storage as storage

        if not storage.project_exists(project_name):
            return 0

        try:
            project_data = storage.load_project_meta(project_name)
        except FileNotFoundError:
            return 0

        seed_state = project_data.get('seed_state', {})
        probe_queue = seed_state.get('probe_queue', [])

        # Find targets without embeddings
        unembedded = []
        unembedded_indices = []

        for i, target in enumerate(probe_queue):
            if isinstance(target, dict):
                embedding = target.get('embedding', [])
            else:
                embedding = getattr(target, 'embedding', [])

            if not embedding:
                unembedded.append(target)
                unembedded_indices.append(i)

            if len(unembedded) >= self.batch_size:
                break

        if not unembedded:
            self._log("All targets have embeddings")
            return 0

        self._log(f"Embedding {len(unembedded)} targets...")

        # Prepare texts for embedding (identifier + context)
        texts = []
        for target in unembedded:
            if isinstance(target, dict):
                identifier = target.get('identifier', '')
                context = target.get('context', '')
            else:
                identifier = target.identifier
                context = target.context

            # Combine identifier and context for richer embedding
            text = f"{identifier}\n\n{context}" if context else identifier
            texts.append(text)

        # Generate embeddings
        try:
            embeddings = embed_texts(texts, model=self.embedding_model)
        except Exception as e:
            self._log(f"Embedding failed: {e}")
            return 0

        # Update targets with embeddings
        updated = 0
        for i, embedding in enumerate(embeddings):
            if embedding:
                idx = unembedded_indices[i]
                if isinstance(probe_queue[idx], dict):
                    probe_queue[idx]['embedding'] = embedding
                else:
                    probe_queue[idx].embedding = embedding
                updated += 1

        # Save back
        if updated:
            seed_state['probe_queue'] = probe_queue
            project_data['seed_state'] = seed_state
            storage.save_project_meta(project_name, project_data)
            self._log(f"Embedded {updated} targets")

        return updated


class SemanticChunkerWorker(BaseWorker):
    """Worker that chunks seed source into semantic units and creates probe targets.

    Unlike seed_explorer which extracts identifiers (function names, class names),
    this worker creates chunks of actual content for more comprehensive probing.
    """

    name = "SEMANTIC-CHUNKER"

    def __init__(self, interval: float = 120.0):
        super().__init__(interval)
        self.max_chunks_per_tick = 50

    def _do_work(self, project_name: str) -> int:
        """Chunk seed content and create probe targets. Returns count of new chunks."""
        if not project_name:
            return 0

        from routes import project_storage as storage

        if not storage.project_exists(project_name):
            return 0

        try:
            project_data = storage.load_project_meta(project_name)
        except FileNotFoundError:
            return 0

        seed = project_data.get('seed', {})
        if not seed or seed.get('type') == 'none' or not seed.get('value'):
            return 0

        seed_state = project_data.get('seed_state', {})
        chunked_files = set(seed_state.get('chunked_files', []))
        probe_queue = seed_state.get('probe_queue', [])

        # Build set of existing identifiers
        existing = {
            t.get('identifier') if isinstance(t, dict) else t.identifier
            for t in probe_queue
        }

        new_chunks = []
        content_type = seed.get('content_type', 'auto')

        if seed.get('type') == 'path':
            path = seed.get('value', '')
            if not os.path.exists(path):
                return 0

            if content_type == 'auto':
                content_type = detect_content_type(path)

            # Find files to chunk
            files_to_chunk = []
            if os.path.isfile(path):
                if path not in chunked_files:
                    files_to_chunk.append(path)
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for f in files:
                        fp = os.path.join(root, f)
                        if fp not in chunked_files:
                            # Filter by content type
                            ext = os.path.splitext(f)[1].lower()
                            if self._should_chunk(ext, content_type):
                                files_to_chunk.append(fp)

            # Chunk files
            for fp in files_to_chunk[:10]:  # Limit per tick
                try:
                    with open(fp, 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()
                except Exception:
                    continue

                # Get chunks based on content type
                ext = os.path.splitext(fp)[1].lower()
                if ext in ('.py',):
                    chunks = chunk_code(content, 'python')
                elif ext in ('.js', '.ts', '.tsx', '.jsx'):
                    chunks = chunk_code(content, 'javascript')
                else:
                    chunks = chunk_text(content, chunk_size=1500, overlap=200)

                for i, chunk in enumerate(chunks):
                    # Create identifier from file + chunk index
                    identifier = f"{os.path.basename(fp)}:chunk_{i}"
                    if identifier in existing:
                        continue

                    new_chunks.append(ProbeTarget(
                        identifier=identifier,
                        source_file=fp,
                        context=chunk[:2000],  # Store chunk as context
                        probed=False,
                        hit=False,
                        hit_count=0
                    ))
                    existing.add(identifier)

                    if len(new_chunks) >= self.max_chunks_per_tick:
                        break

                chunked_files.add(fp)

                if len(new_chunks) >= self.max_chunks_per_tick:
                    break

        elif seed.get('type') == 'content':
            # Chunk raw content
            content = seed.get('value', '')
            if 'raw_content' not in chunked_files:
                chunks = chunk_text(content, chunk_size=1500, overlap=200)
                for i, chunk in enumerate(chunks):
                    identifier = f"content:chunk_{i}"
                    if identifier in existing:
                        continue

                    new_chunks.append(ProbeTarget(
                        identifier=identifier,
                        source_file="",
                        context=chunk[:2000],
                        probed=False,
                        hit=False,
                        hit_count=0
                    ))

                chunked_files.add('raw_content')

        # Add new chunks to queue
        if new_chunks:
            for chunk in new_chunks:
                probe_queue.append(chunk.model_dump())

            seed_state['probe_queue'] = probe_queue
            seed_state['chunked_files'] = list(chunked_files)
            project_data['seed_state'] = seed_state
            storage.save_project_meta(project_name, project_data)
            self._log(f"Created {len(new_chunks)} semantic chunks")

        return len(new_chunks)

    def _should_chunk(self, ext: str, content_type: str) -> bool:
        """Check if file extension should be chunked based on content type."""
        code_exts = {'.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.kt', '.swift', '.go', '.rs', '.c', '.cpp', '.h'}
        text_exts = {'.txt', '.md', '.rst', '.json', '.yaml', '.yml', '.xml', '.html'}
        legal_exts = {'.txt', '.md', '.pdf', '.doc', '.docx'}

        if content_type == 'code':
            return ext in code_exts
        elif content_type == 'legal':
            return ext in legal_exts
        elif content_type == 'general':
            return ext in code_exts | text_exts
        else:  # auto
            return ext in code_exts | text_exts


# Module-level instances
_indexer: Optional[SemanticIndexerWorker] = None
_chunker: Optional[SemanticChunkerWorker] = None


def get_indexer() -> SemanticIndexerWorker:
    global _indexer
    if _indexer is None:
        _indexer = SemanticIndexerWorker(interval=60.0)
    return _indexer


def get_chunker() -> SemanticChunkerWorker:
    global _chunker
    if _chunker is None:
        _chunker = SemanticChunkerWorker(interval=120.0)
    return _chunker


def start_workers() -> tuple[SemanticIndexerWorker, SemanticChunkerWorker]:
    indexer = get_indexer()
    chunker = get_chunker()
    indexer.start()
    chunker.start()
    return indexer, chunker


def stop_workers() -> None:
    global _indexer, _chunker
    if _indexer:
        _indexer.stop()
        _indexer = None
    if _chunker:
        _chunker.stop()
        _chunker = None
