"""
Web Angles Worker - Background process for discovering new investigation angles.

Searches the web for new angles/leads based on discovered entities,
updates project web_leads metadata.
"""

import threading
import time
from typing import Optional
from dataclasses import dataclass


@dataclass
class WorkerStats:
    """Track worker activity."""
    searches_run: int = 0
    angles_found: int = 0
    errors: int = 0
    last_run: Optional[float] = None
    last_project: Optional[str] = None


class WebAnglesWorker:
    """
    Background worker that discovers new investigation angles via web search.
    """

    def __init__(self, interval: float = 60.0):
        """
        Args:
            interval: Seconds between search cycles
        """
        self.interval = interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.stats = WorkerStats()

    def start(self):
        """Start the worker thread."""
        if self._thread and self._thread.is_alive():
            print("[WEB-ANGLES-WORKER] Already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[WEB-ANGLES-WORKER] Started (interval={self.interval}s)")

    def stop(self):
        """Stop the worker thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("[WEB-ANGLES-WORKER] Stopped")

    def is_running(self) -> bool:
        """Check if worker is active."""
        return self._thread is not None and self._thread.is_alive()

    def _run_loop(self):
        """Main worker loop."""
        from routes import project_storage as storage

        print("[WEB-ANGLES-WORKER] Loop started")

        while not self._stop_event.is_set():
            try:
                projects = storage.list_projects()

                for project in projects:
                    if self._stop_event.is_set():
                        break

                    project_name = project.get("name") if isinstance(project, dict) else project
                    self._search_angles(project_name)

                self.stats.last_run = time.time()

            except Exception as e:
                print(f"[WEB-ANGLES-WORKER] Error in loop: {e}")
                self.stats.errors += 1

            self._stop_event.wait(self.interval)

        print("[WEB-ANGLES-WORKER] Loop ended")

    def _search_angles(self, project_name: str):
        """Search for new angles for a project."""
        from routes import project_storage as storage
        from routes.helpers import search_new_angles
        from collections import Counter

        if not storage.project_exists(project_name):
            return

        try:
            proj = storage.load_project_meta(project_name)
            topic = proj.get("topic", "")

            if not topic:
                return

            # Get top entities from corpus
            corpus = storage.load_corpus(project_name)
            entity_counts = Counter()
            for item in corpus:
                for entity in item.get("entities", []):
                    entity_counts[entity] += 1

            known_entities = [e for e, _ in entity_counts.most_common(20)]

            if not known_entities:
                return

            print(f"[WEB-ANGLES-WORKER] {project_name}: searching angles for {len(known_entities)} entities")
            self.stats.last_project = project_name

            # Search for new angles
            result = search_new_angles(topic, known_entities)

            if "error" not in result:
                self.stats.searches_run += 1

                # Count new angles
                news = result.get("news_angles", [])
                wiki = result.get("wiki_context", [])
                self.stats.angles_found += len(news) + len(wiki)

                # Save to project
                proj["web_leads"] = result
                proj["web_leads_updated"] = time.time()
                storage.save_project_meta(project_name, proj)

                print(f"[WEB-ANGLES-WORKER] {project_name}: found {len(news)} news, {len(wiki)} wiki angles")

        except Exception as e:
            print(f"[WEB-ANGLES-WORKER] Failed for {project_name}: {e}")
            self.stats.errors += 1


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
