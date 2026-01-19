"""
Interrogator Worker - Synthesizes findings into working theory.

Runs periodically to update the narrative based on corpus data.
"""

from typing import Optional
from collections import Counter

from .base import BaseWorker


class InterrogatorWorker(BaseWorker):
    """
    Periodically synthesizes corpus data into working theory.

    This is the "theory writer" side of the dialectic.
    The skeptic worker challenges, this worker defends and refines.
    """

    def __init__(self, interval: float = 60.0):
        super().__init__(name="INTERROGATOR-WORKER", interval=interval)

    def _do_work(self, project_name: str) -> int:
        """Synthesize findings for a project."""
        from routes import project_storage as storage
        from routes.synthesis import synthesize_narrative

        if not storage.project_exists(project_name):
            return 0

        proj = storage.load_project_meta(project_name)
        topic = proj.get("topic", "")

        if not topic:
            return 0

        # Get entities from corpus
        corpus = storage.load_corpus(project_name)
        if not corpus:
            return 0

        entity_counts = Counter()
        for item in corpus:
            for entity in item.get("entities", []):
                entity_counts[entity] += 1

        # Need at least some entities to synthesize
        if len(entity_counts) < 3:
            return 0

        # Build scored entity list (entity, score, freq)
        top_entities = [
            (entity, count, count)
            for entity, count in entity_counts.most_common(20)
        ]

        research_context = proj.get("research_context", "")
        user_notes = proj.get("user_notes", "")

        print(f"[{self.name}] Synthesizing {project_name} ({len(top_entities)} entities)")

        narrative = synthesize_narrative(
            project_name=project_name,
            topic=topic,
            entities=top_entities,
            research_context=research_context,
            user_notes=user_notes,
            storage=storage
        )

        if narrative:
            print(f"[{self.name}] Done - {len(narrative)} chars")
            return 1
        else:
            print(f"[{self.name}] Synthesis returned None")
            return 0


# Global instance
_worker: Optional[InterrogatorWorker] = None


def get_worker() -> InterrogatorWorker:
    global _worker
    if _worker is None:
        _worker = InterrogatorWorker()
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
