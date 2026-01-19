"""
Seed Explorer Worker - progressively extracts probe targets from seed source.

On each tick:
1. Check if probe_queue is low
2. If so, extract more identifiers from seed
3. Prioritize hot_zones (areas with previous hits)
4. Add new targets to probe_queue
"""

import os
from typing import Optional
from .base import BaseWorker
from .seed_extractor import extract_from_path, extract_from_content, detect_content_type
from models import Project, ProbeTarget


class SeedExplorerWorker(BaseWorker):
    """Worker that progressively extracts probe targets from seed source."""

    name = "SEED-EXPLORER"

    def __init__(self, interval: float = 30.0):
        super().__init__(interval)
        self.targets_per_batch = 10  # How many targets to extract per tick
        self.min_queue_size = 5  # Extract more when queue drops below this

    def _do_work(self, project_name: str) -> int:
        """Extract probe targets from seed source. Returns count of new targets."""
        if not project_name:
            return 0

        from routes import project_storage as storage

        if not storage.project_exists(project_name):
            return 0

        try:
            project_data = storage.load_project_meta(project_name)
        except FileNotFoundError:
            return 0

        # Convert dict to model-like object for easier access
        class ProjectWrapper:
            def __init__(self, data):
                self._data = data
                self.seed = type('Seed', (), data.get('seed', {'type': 'none', 'value': '', 'content_type': 'auto'}))()
                self.seed_state = type('SeedState', (), {
                    'explored_paths': data.get('seed_state', {}).get('explored_paths', []),
                    'probe_queue': [ProbeTarget(**t) if isinstance(t, dict) else t for t in data.get('seed_state', {}).get('probe_queue', [])],
                    'probed': [ProbeTarget(**t) if isinstance(t, dict) else t for t in data.get('seed_state', {}).get('probed', [])],
                    'hot_zones': data.get('seed_state', {}).get('hot_zones', [])
                })()

            def to_dict(self):
                return {
                    **self._data,
                    'seed': {'type': self.seed.type, 'value': self.seed.value, 'content_type': self.seed.content_type},
                    'seed_state': {
                        'explored_paths': self.seed_state.explored_paths,
                        'probe_queue': [t.model_dump() if hasattr(t, 'model_dump') else t for t in self.seed_state.probe_queue],
                        'probed': [t.model_dump() if hasattr(t, 'model_dump') else t for t in self.seed_state.probed],
                        'hot_zones': self.seed_state.hot_zones
                    }
                }

        project = ProjectWrapper(project_data)

        # Check if we have a seed source
        if not project.seed or project.seed.type == 'none' or not project.seed.value:
            return 0  # No seed configured

        # Check if queue needs refilling
        queue_size = len(project.seed_state.probe_queue)
        if queue_size >= self.min_queue_size:
            self._log(f"Queue has {queue_size} targets, skipping extraction")
            return 0

        # Extract more targets
        new_targets = self._extract_batch(project)

        if new_targets:
            project.seed_state.probe_queue.extend(new_targets)
            storage.save_project_meta(project_name, project.to_dict())
            self._log(f"Added {len(new_targets)} targets to queue (total: {len(project.seed_state.probe_queue)})")
            return len(new_targets)

        return 0

    def _extract_batch(self, project) -> list[ProbeTarget]:
        """Extract a batch of new probe targets from seed."""
        seed = project.seed
        state = project.seed_state

        # Build set of already-seen identifiers
        seen_identifiers = set()
        for t in state.probe_queue:
            seen_identifiers.add(t.identifier)
        for t in state.probed:
            seen_identifiers.add(t.identifier)

        new_targets = []
        content_type = seed.content_type

        if seed.type == 'path':
            if not os.path.exists(seed.value):
                self._log(f"Seed path not found: {seed.value}")
                return []

            if content_type == 'auto':
                content_type = detect_content_type(seed.value)

            # Prioritize hot zones first
            paths_to_explore = []
            if state.hot_zones:
                for hz in state.hot_zones:
                    full_path = os.path.join(seed.value, hz) if not os.path.isabs(hz) else hz
                    if os.path.exists(full_path):
                        paths_to_explore.append(full_path)

            # Then the main path
            paths_to_explore.append(seed.value)

            explored_set = set(state.explored_paths)

            for path in paths_to_explore:
                if len(new_targets) >= self.targets_per_batch:
                    break

                for target in extract_from_path(path, content_type, max_per_file=5):
                    if target.identifier in seen_identifiers:
                        continue

                    seen_identifiers.add(target.identifier)
                    new_targets.append(target)

                    # Track explored files
                    if target.source_file and target.source_file not in explored_set:
                        state.explored_paths.append(target.source_file)
                        explored_set.add(target.source_file)

                    if len(new_targets) >= self.targets_per_batch:
                        break

        elif seed.type == 'content':
            # Extract from raw pasted content
            for target in extract_from_content(seed.value, content_type, max_count=self.targets_per_batch):
                if target.identifier in seen_identifiers:
                    continue
                seen_identifiers.add(target.identifier)
                new_targets.append(target)
                if len(new_targets) >= self.targets_per_batch:
                    break

        elif seed.type == 'url':
            # TODO: Fetch URL content and extract
            self._log(f"URL extraction not yet implemented: {seed.value}")

        return new_targets


# Module-level worker instance
_worker: Optional[SeedExplorerWorker] = None


def get_worker() -> SeedExplorerWorker:
    global _worker
    if _worker is None:
        _worker = SeedExplorerWorker(interval=30.0)
    return _worker


def start_worker() -> SeedExplorerWorker:
    worker = get_worker()
    worker.start()
    return worker


def stop_worker() -> None:
    global _worker
    if _worker:
        _worker.stop()
        _worker = None
