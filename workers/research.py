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
    Background worker that continuously researches entities in the active project.

    Runs at a low clip rate to avoid overwhelming APIs while keeping
    research context fresh.
    """

    def __init__(self, interval: float = 20.0, max_queries_per_run: int = 10):
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
        self._active_project: Optional[str] = None  # Only research this project

    def set_project(self, project_name: Optional[str]):
        """Set the active project to research. None to pause."""
        self._active_project = project_name
        if project_name:
            print(f"[RESEARCH-WORKER] Now focused on: {project_name}")

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
        from routes import project_storage as storage

        print("[RESEARCH-WORKER] Loop started")

        while not self._stop_event.is_set():
            try:
                project = self._active_project

                # If no active project, pick most recently updated
                if not project:
                    projects = storage.list_projects()
                    if projects:
                        # list_projects returns sorted by updated desc
                        first = projects[0]
                        project = first.get("name") if isinstance(first, dict) else first

                if project:
                    self._research_project(project)
                    self.stats.last_run = time.time()

            except Exception as e:
                print(f"[RESEARCH-WORKER] Error in loop: {e}")
                self.stats.errors += 1

            # Wait for next cycle
            self._stop_event.wait(self.interval)

        print("[RESEARCH-WORKER] Loop ended")

    def _research_project(self, project_name: str):
        """Research entities in a single project using LLM-directed queries."""
        from routes import project_storage as storage

        if not storage.project_exists(project_name):
            return

        try:
            proj = storage.load_project_meta(project_name)
            topic = proj.get("topic", "")

            if not topic:
                return

            # Get entities to research
            entities_to_research = self._get_research_targets(project_name, proj)
            existing_research = proj.get("research_context", "")

            print(f"[RESEARCH-WORKER] {project_name}: researching with {len(entities_to_research)} known entities")
            self.stats.last_project = project_name

            # LLM-directed: Get smart search queries
            from routes.analyze.research import research
            from routes.analyze.research.cleaner import suggest_next_searches, is_useful_research

            # Ask LLM what to search for
            queries = suggest_next_searches(topic, entities_to_research, existing_research)
            if not queries:
                # Fallback to simple entity queries
                topic_words = topic.split()[:2]
                topic_prefix = " ".join(topic_words)
                queries = [f"{topic_prefix} {e}" for e in entities_to_research[:5]]

            print(f"[RESEARCH-WORKER] LLM suggested queries: {queries[:3]}...")

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

                    self.stats.queries_run += 1

                    # Filter with LLM - only keep useful results
                    useful_docs = []
                    for doc in result.documents:
                        is_useful, reason = is_useful_research(doc.content, doc.title, topic)
                        if is_useful:
                            useful_docs.append(doc)
                            print(f"[RESEARCH-WORKER] USEFUL: {doc.title[:40]}... - {reason[:50]}")
                        else:
                            print(f"[RESEARCH-WORKER] TRASH: {doc.title[:40]}... - {reason[:50]}")

                    self.stats.documents_fetched += len(useful_docs)
                    self.stats.documents_cached += result.cached_count
                    docs_found += len(useful_docs)

                    if useful_docs:
                        # Update project research context with useful docs only
                        result.documents = useful_docs
                        self._update_project_research(project_name, result)

                        self._notify('research_update', {
                            'project': project_name,
                            'query': query,
                            'documents_found': len(useful_docs)
                        })

                except Exception as e:
                    print(f"[RESEARCH-WORKER] Query failed for '{query[:50]}': {e}")
                    self.stats.errors += 1

                # Small delay between queries
                time.sleep(1)

        except Exception as e:
            print(f"[RESEARCH-WORKER] Failed to research {project_name}: {e}")
            self.stats.errors += 1

    def _get_research_targets(self, project_name: str, proj: dict) -> list[str]:
        """Get list of entities that need research."""
        targets = []

        # Priority 1: PRIVATE/unverified entities (most valuable)
        verification = proj.get("entity_verification", {})
        unverified = verification.get("unverified", [])
        for item in unverified[:5]:
            entity = item.get("entity") if isinstance(item, dict) else item
            if entity:
                targets.append(entity)

        # Priority 2: Top entities from corpus
        # (These are already in the word cloud, so researching them adds context)
        if len(targets) < 5:
            from routes import project_storage as storage
            from collections import Counter
            try:
                corpus = storage.load_corpus(project_name)
                entity_counts = Counter()
                for item in corpus:
                    for entity in item.get("entities", []):
                        entity_counts[entity] += 1

                # Get top entities by frequency
                for entity, count in entity_counts.most_common(10):
                    if entity not in targets and len(targets) < 5:
                        targets.append(entity)
            except Exception:
                pass

        return targets

    def _update_project_research(self, project_name: str, result):
        """Update project with new research context."""
        from routes import project_storage as storage

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
