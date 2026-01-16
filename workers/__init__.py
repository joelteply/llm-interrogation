"""
Background workers - continuous processing.

All workers extend BaseWorker which handles:
- Thread lifecycle
- Stats tracking via shared WorkerStats model
- Project focus
- Callbacks

Workers:
- ResearchWorker: LLM-directed document research
- WebAnglesWorker: Web angle discovery
- EntityVerificationWorker: PUBLIC/PRIVATE classification
- SkepticWorker: Devil's advocate analysis
"""

from .base import BaseWorker
from .research import ResearchWorker
from .web_angles import WebAnglesWorker
from .entity_verification import EntityVerificationWorker
from .skeptic import SkepticWorker

__all__ = [
    'BaseWorker',
    'ResearchWorker',
    'WebAnglesWorker',
    'EntityVerificationWorker',
    'SkepticWorker',
]


def start_all_workers():
    """Start all background workers."""
    from .research import start_worker as start_research
    from .web_angles import start_worker as start_angles
    from .entity_verification import start_worker as start_verify
    from .skeptic import start_worker as start_skeptic

    start_research()
    start_angles()
    start_verify()
    start_skeptic()


def stop_all_workers():
    """Stop all background workers."""
    from .research import stop_worker as stop_research
    from .web_angles import stop_worker as stop_angles
    from .entity_verification import stop_worker as stop_verify
    from .skeptic import stop_worker as stop_skeptic

    stop_research()
    stop_angles()
    stop_verify()
    stop_skeptic()


def get_all_stats() -> dict:
    """Get stats from all workers using unified format."""
    from .research import get_worker as get_research
    from .web_angles import get_worker as get_angles
    from .entity_verification import get_worker as get_verify
    from .skeptic import get_worker as get_skeptic

    return {
        "research": get_research().get_stats(),
        "web_angles": get_angles().get_stats(),
        "entity_verification": get_verify().get_stats(),
        "skeptic": get_skeptic().get_stats(),
    }


def set_all_project(project_name: str):
    """Set active project for all workers."""
    from .research import get_worker as get_research
    from .skeptic import get_worker as get_skeptic

    get_research().set_project(project_name)
    get_skeptic().set_project(project_name)
