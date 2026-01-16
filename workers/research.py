"""
Research Worker - LLM-directed document research.
"""

import time
from typing import Optional

from .base import BaseWorker


class ResearchWorker(BaseWorker):
    """
    Continuously researches entities using LLM-directed queries.

    Uses LLM to suggest smart search queries, fetches documents,
    filters out garbage with LLM evaluation.
    """

    def __init__(self, interval: float = 20.0, max_queries_per_run: int = 10):
        super().__init__(name="RESEARCH-WORKER", interval=interval)
        self.max_queries_per_run = max_queries_per_run

    def _do_work(self, project_name: str) -> int:
        """Research entities in a project."""
        from routes import project_storage as storage

        if not storage.project_exists(project_name):
            return 0

        proj = storage.load_project_meta(project_name)
        topic = proj.get("topic", "")
        if not topic:
            return 0

        # Get entities to research
        entities = self._get_research_targets(project_name, proj)
        existing_research = proj.get("research_context", "")

        print(f"[{self.name}] {project_name}: {len(entities)} known entities")

        # LLM-directed queries
        from routes.analyze.research import research
        from routes.analyze.research.cleaner import suggest_next_searches, is_useful_research

        queries = suggest_next_searches(topic, entities, existing_research)
        if not queries:
            topic_words = topic.split()[:2]
            topic_prefix = " ".join(topic_words)
            queries = [f"{topic_prefix} {e}" for e in entities[:5]]

        print(f"[{self.name}] Queries: {queries[:3]}...")

        docs_found = 0
        for query in queries[:self.max_queries_per_run]:
            if self._stop_event.is_set():
                break

            try:
                result = research(
                    query=query,
                    project_name=project_name,
                    sources=['documentcloud', 'web'],
                    max_per_source=5
                )

                # Filter with LLM
                useful_docs = []
                for doc in result.documents:
                    is_useful, reason = is_useful_research(doc.content, doc.title, topic)
                    if is_useful:
                        useful_docs.append(doc)
                        print(f"[{self.name}] USEFUL: {doc.title[:40]}...")
                    else:
                        print(f"[{self.name}] TRASH: {doc.title[:40]}...")

                docs_found += len(useful_docs)
                self.stats.items_cached += result.cached_count

                if useful_docs:
                    result.documents = useful_docs
                    self._update_project_research(project_name, result)
                    self.notify('research_update', {
                        'project': project_name,
                        'query': query,
                        'documents_found': len(useful_docs)
                    })

            except Exception as e:
                print(f"[{self.name}] Query failed: {e}")
                self.stats.errors += 1

            time.sleep(1)

        return docs_found

    def _get_research_targets(self, project_name: str, proj: dict) -> list[str]:
        """Get entities that need research."""
        targets = []

        # Priority 1: PRIVATE/unverified entities
        verification = proj.get("entity_verification", {})
        unverified = verification.get("unverified", [])
        for item in unverified[:5]:
            entity = item.get("entity") if isinstance(item, dict) else item
            if entity:
                targets.append(entity)

        # Priority 2: Top corpus entities
        if len(targets) < 5:
            from routes import project_storage as storage
            from collections import Counter
            try:
                corpus = storage.load_corpus(project_name)
                entity_counts = Counter()
                for item in corpus:
                    for entity in item.get("entities", []):
                        entity_counts[entity] += 1

                for entity, _ in entity_counts.most_common(10):
                    if entity not in targets and len(targets) < 5:
                        targets.append(entity)
            except Exception:
                pass

        return targets

    def _update_project_research(self, project_name: str, result) -> None:
        """Update project with research context."""
        from routes import project_storage as storage

        try:
            proj = storage.load_project_meta(project_name)
            existing = proj.get("research_context", "")
            new_content = result.raw_content[:5000]

            if new_content and new_content not in existing:
                proj["research_context"] = (existing + "\n\n---\n\n" + new_content)[-20000:]
                proj["research_updated"] = time.time()
                storage.save_project_meta(project_name, proj)

        except Exception as e:
            print(f"[{self.name}] Update failed: {e}")


# Global instance
_worker: Optional[ResearchWorker] = None


def get_worker() -> ResearchWorker:
    global _worker
    if _worker is None:
        _worker = ResearchWorker()
    return _worker


def start_worker():
    worker = get_worker()
    if not worker.is_running():
        worker.start()
    return worker


def stop_worker():
    global _worker
    if _worker:
        _worker.stop()
