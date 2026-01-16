"""
Research Worker - Background process for continuous document research.

Polls projects for new entities to research, fetches from DocumentCloud/web,
and updates project metadata (which probe reads from cache).
"""

import threading
import time
import os
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class WorkerStats:
    """Track worker activity."""
    queries_run: int = 0
    documents_fetched: int = 0
    documents_cached: int = 0
    errors: int = 0
    last_run: Optional[float] = None
    last_project: Optional[str] = None


class ResearchWorker:
    """
    Background worker that continuously researches entities in projects.

    Runs at a low clip rate to avoid overwhelming APIs while keeping
    research context fresh.
    """

    def __init__(self, interval: float = 30.0, max_queries_per_run: int = 3):
        """
        Args:
            interval: Seconds between research cycles
            max_queries_per_run: Max queries per project per cycle
        """
        self.interval = interval
        self.max_queries_per_run = max_queries_per_run
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.stats = WorkerStats()
        self._callbacks: list = []  # SSE callbacks to notify

    def start(self):
        """Start the worker thread."""
        if self._thread and self._thread.is_alive():
            print("[RESEARCH-WORKER] Already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[RESEARCH-WORKER] Started (interval={self.interval}s)")

    def stop(self):
        """Stop the worker thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("[RESEARCH-WORKER] Stopped")

    def is_running(self) -> bool:
        """Check if worker is active."""
        return self._thread is not None and self._thread.is_alive()

    def add_callback(self, callback):
        """Add SSE callback to notify of research updates."""
        self._callbacks.append(callback)

    def remove_callback(self, callback):
        """Remove SSE callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify(self, event_type: str, data: dict):
        """Notify all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(event_type, data)
            except Exception as e:
                print(f"[RESEARCH-WORKER] Callback error: {e}")

    def _run_loop(self):
        """Main worker loop."""
        from storage import storage

        print("[RESEARCH-WORKER] Loop started")

        while not self._stop_event.is_set():
            try:
                # Get all projects
                projects = storage.list_projects()

                for project_name in projects:
                    if self._stop_event.is_set():
                        break

                    self._research_project(project_name)

                self.stats.last_run = time.time()

            except Exception as e:
                print(f"[RESEARCH-WORKER] Error in loop: {e}")
                self.stats.errors += 1

            # Wait for next cycle
            self._stop_event.wait(self.interval)

        print("[RESEARCH-WORKER] Loop ended")

    def _research_project(self, project_name: str):
        """Research entities in a single project."""
        from storage import storage

        if not storage.project_exists(project_name):
            return

        try:
            proj = storage.load_project_meta(project_name)
            topic = proj.get("topic", "")

            if not topic:
                return

            # Get entities to research
            entities_to_research = self._get_research_targets(proj)

            if not entities_to_research:
                return

            print(f"[RESEARCH-WORKER] {project_name}: researching {len(entities_to_research)} entities")
            self.stats.last_project = project_name

            # Run research queries
            from routes.analyze.research import research

            for entity in entities_to_research[:self.max_queries_per_run]:
                if self._stop_event.is_set():
                    break

                query = f"{topic} {entity}"

                try:
                    result = research(
                        query=query,
                        project_name=project_name,
                        sources=['documentcloud', 'web'],
                        max_per_source=3
                    )

                    self.stats.queries_run += 1
                    self.stats.documents_fetched += result.fetched_count
                    self.stats.documents_cached += result.cached_count

                    if result.documents:
                        # Update project research context
                        self._update_project_research(project_name, result)

                        # Notify callbacks
                        self._notify('research_update', {
                            'project': project_name,
                            'entity': entity,
                            'documents_found': len(result.documents)
                        })

                except Exception as e:
                    print(f"[RESEARCH-WORKER] Query failed for '{entity}': {e}")
                    self.stats.errors += 1

                # Small delay between queries
                time.sleep(1)

        except Exception as e:
            print(f"[RESEARCH-WORKER] Failed to research {project_name}: {e}")
            self.stats.errors += 1

    def _get_research_targets(self, proj: dict) -> list[str]:
        """Get list of entities that need research."""
        targets = []

        # Priority 1: PRIVATE/unverified entities (most valuable)
        verification = proj.get("entity_verification", {})
        unverified = verification.get("unverified", [])
        for item in unverified[:5]:
            entity = item.get("entity") if isinstance(item, dict) else item
            if entity:
                targets.append(entity)

        # Priority 2: Top entities from findings
        # (These are already in the word cloud, so researching them adds context)
        if len(targets) < 5:
            # Load findings to get top entities
            from storage import storage
            try:
                findings_data = storage.load_findings(proj.get("name", ""))
                if findings_data:
                    entities = findings_data.get("entities", {})
                    sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)
                    for entity, count in sorted_entities[:10]:
                        if entity not in targets and len(targets) < 5:
                            targets.append(entity)
            except Exception:
                pass

        return targets

    def _update_project_research(self, project_name: str, result):
        """Update project with new research context."""
        from storage import storage

        try:
            proj = storage.load_project_meta(project_name)

            # Append to existing research context
            existing = proj.get("research_context", "")
            new_content = result.raw_content[:5000]  # Limit size

            if new_content and new_content not in existing:
                # Append with separator
                proj["research_context"] = (existing + "\n\n---\n\n" + new_content)[-20000:]  # Keep last 20k chars
                proj["research_updated"] = time.time()
                storage.save_project_meta(project_name, proj)

        except Exception as e:
            print(f"[RESEARCH-WORKER] Failed to update {project_name}: {e}")


# Global worker instance
_worker: Optional[ResearchWorker] = None


def get_worker() -> ResearchWorker:
    """Get or create the global research worker."""
    global _worker
    if _worker is None:
        _worker = ResearchWorker()
    return _worker


def start_worker():
    """Start the global research worker."""
    worker = get_worker()
    if not worker.is_running():
        worker.start()
    return worker


def stop_worker():
    """Stop the global research worker."""
    global _worker
    if _worker:
        _worker.stop()
