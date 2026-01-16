"""
Web Angles Worker - Discovers new investigation angles via web search.
"""

import time
from typing import Optional
from collections import Counter

from .base import BaseWorker


class WebAnglesWorker(BaseWorker):
    """
    Discovers new investigation angles via web search.
    """

    def __init__(self, interval: float = 60.0):
        super().__init__(name="WEB-ANGLES-WORKER", interval=interval)

    def _do_work(self, project_name: str) -> int:
        """Search for new angles."""
        from routes import project_storage as storage
        from routes.helpers import search_new_angles

        if not storage.project_exists(project_name):
            return 0

        proj = storage.load_project_meta(project_name)
        topic = proj.get("topic", "")
        if not topic:
            return 0

        # Get top entities
        corpus = storage.load_corpus(project_name)
        entity_counts = Counter()
        for item in corpus:
            for entity in item.get("entities", []):
                entity_counts[entity] += 1

        known_entities = [e for e, _ in entity_counts.most_common(20)]
        if not known_entities:
            return 0

        print(f"[{self.name}] {project_name}: {len(known_entities)} entities")

        result = search_new_angles(topic, known_entities)

        if "error" in result:
            return 0

        news = result.get("news_angles", [])
        wiki = result.get("wiki_context", [])

        proj["web_leads"] = result
        proj["web_leads_updated"] = time.time()
        storage.save_project_meta(project_name, proj)

        print(f"[{self.name}] {project_name}: {len(news)} news, {len(wiki)} wiki")

        return len(news) + len(wiki)


# Global instance
_worker: Optional[WebAnglesWorker] = None


def get_worker() -> WebAnglesWorker:
    global _worker
    if _worker is None:
        _worker = WebAnglesWorker()
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
