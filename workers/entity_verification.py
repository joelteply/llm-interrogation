"""
Entity Verification Worker - Background process for verifying entities as PUBLIC/PRIVATE.

Checks discovered entities against web search to determine if they're
publicly known or potentially private/leaked information.
"""

import threading
import time
from typing import Optional
from dataclasses import dataclass


@dataclass
class WorkerStats:
    """Track worker activity."""
    entities_verified: int = 0
    public_found: int = 0
    private_found: int = 0
    errors: int = 0
    last_run: Optional[float] = None
    last_project: Optional[str] = None


class EntityVerificationWorker:
    """
    Background worker that verifies entities as PUBLIC vs PRIVATE.

    PUBLIC = Found in public web sources (Wikipedia, news, etc.)
    PRIVATE = Not found publicly - potentially leaked/insider info
    """

    def __init__(self, interval: float = 45.0, max_entities_per_run: int = 10):
        """
        Args:
            interval: Seconds between verification cycles
            max_entities_per_run: Max entities to verify per project per cycle
        """
        self.interval = interval
        self.max_entities_per_run = max_entities_per_run
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.stats = WorkerStats()

    def start(self):
        """Start the worker thread."""
        if self._thread and self._thread.is_alive():
            print("[VERIFY-WORKER] Already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[VERIFY-WORKER] Started (interval={self.interval}s)")

    def stop(self):
        """Stop the worker thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("[VERIFY-WORKER] Stopped")

    def is_running(self) -> bool:
        """Check if worker is active."""
        return self._thread is not None and self._thread.is_alive()

    def _run_loop(self):
        """Main worker loop."""
        from routes import project_storage as storage

        print("[VERIFY-WORKER] Loop started")

        while not self._stop_event.is_set():
            try:
                projects = storage.list_projects()

                for project in projects:
                    if self._stop_event.is_set():
                        break

                    project_name = project.get("name") if isinstance(project, dict) else project
                    self._verify_project_entities(project_name)

                self.stats.last_run = time.time()

            except Exception as e:
                print(f"[VERIFY-WORKER] Error in loop: {e}")
                self.stats.errors += 1

            self._stop_event.wait(self.interval)

        print("[VERIFY-WORKER] Loop ended")

    def _verify_project_entities(self, project_name: str):
        """Verify entities for a project."""
        from routes import project_storage as storage
        from routes.helpers import verify_entities
        from collections import Counter

        if not storage.project_exists(project_name):
            return

        try:
            proj = storage.load_project_meta(project_name)
            topic = proj.get("topic", "")

            if not topic:
                return

            # Get existing verification data
            existing = proj.get("entity_verification", {})
            already_verified = set()
            for item in existing.get("verified", []):
                entity = item.get("entity") if isinstance(item, dict) else item
                if entity:
                    already_verified.add(entity.lower())
            for item in existing.get("unverified", []):
                entity = item.get("entity") if isinstance(item, dict) else item
                if entity:
                    already_verified.add(entity.lower())

            # Get top entities from corpus that aren't verified yet
            corpus = storage.load_corpus(project_name)
            entity_counts = Counter()
            for item in corpus:
                for entity in item.get("entities", []):
                    if entity.lower() not in already_verified:
                        entity_counts[entity] += 1

            # Get top unverified entities
            to_verify = [e for e, _ in entity_counts.most_common(self.max_entities_per_run)]

            if not to_verify:
                return

            print(f"[VERIFY-WORKER] {project_name}: verifying {len(to_verify)} entities")
            self.stats.last_project = project_name

            # Verify entities
            result = verify_entities(to_verify, topic, max_entities=self.max_entities_per_run)

            if "error" not in result:
                self.stats.entities_verified += len(to_verify)

                # Merge with existing
                verified = existing.get("verified", [])
                unverified = existing.get("unverified", [])

                for item in result.get("verified", []):
                    verified.append(item)
                    self.stats.public_found += 1

                for item in result.get("unverified", []):
                    unverified.append(item)
                    self.stats.private_found += 1

                # Save updated verification
                proj["entity_verification"] = {
                    "verified": verified[-100:],  # Keep last 100
                    "unverified": unverified[-100:],
                    "updated": time.time()
                }
                storage.save_project_meta(project_name, proj)

                print(f"[VERIFY-WORKER] {project_name}: {len(result.get('verified', []))} public, {len(result.get('unverified', []))} private")

        except Exception as e:
            print(f"[VERIFY-WORKER] Failed for {project_name}: {e}")
            self.stats.errors += 1


# Global instance
_worker: Optional[EntityVerificationWorker] = None


def get_worker() -> EntityVerificationWorker:
    global _worker
    if _worker is None:
        _worker = EntityVerificationWorker()
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
