"""
Entity Verification Worker - PUBLIC/PRIVATE classification.
"""

import time
from typing import Optional
from collections import Counter

from .base import BaseWorker


class EntityVerificationWorker(BaseWorker):
    """
    Verifies entities as PUBLIC (found in web) vs PRIVATE (not found).
    """

    def __init__(self, interval: float = 45.0, max_entities_per_run: int = 10):
        super().__init__(name="VERIFY-WORKER", interval=interval)
        self.max_entities_per_run = max_entities_per_run

    def _do_work(self, project_name: str) -> int:
        """Verify entities for a project."""
        from routes import project_storage as storage
        from routes.helpers import verify_entities

        if not storage.project_exists(project_name):
            return 0

        proj = storage.load_project_meta(project_name)
        topic = proj.get("topic", "")
        if not topic:
            return 0

        # Get already verified
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

        # Get unverified entities from corpus
        corpus = storage.load_corpus(project_name)
        entity_counts = Counter()
        for item in corpus:
            for entity in item.get("entities", []):
                if entity.lower() not in already_verified:
                    entity_counts[entity] += 1

        to_verify = [e for e, _ in entity_counts.most_common(self.max_entities_per_run)]
        if not to_verify:
            return 0

        print(f"[{self.name}] {project_name}: verifying {len(to_verify)}")

        result = verify_entities(to_verify, topic, max_entities=self.max_entities_per_run)

        if "error" in result:
            return 0

        # Merge results
        verified = existing.get("verified", [])
        unverified = existing.get("unverified", [])

        public_count = 0
        private_count = 0

        for item in result.get("verified", []):
            verified.append(item)
            public_count += 1

        for item in result.get("unverified", []):
            unverified.append(item)
            private_count += 1

        proj["entity_verification"] = {
            "verified": verified[-100:],
            "unverified": unverified[-100:],
            "updated": time.time()
        }
        storage.save_project_meta(project_name, proj)

        print(f"[{self.name}] {project_name}: {public_count} public, {private_count} private")

        return public_count + private_count


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
