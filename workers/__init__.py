"""
Background workers for non-blocking operations.

Workers run continuously and poll for work, updating project metadata
which the main probe reads from cache.

Workers:
- ResearchWorker: Fetches docs from DocumentCloud/web for entities
- WebAnglesWorker: Discovers new investigation angles via web search
- EntityVerificationWorker: Verifies entities as PUBLIC vs PRIVATE
"""

from .research import ResearchWorker
from .web_angles import WebAnglesWorker
from .entity_verification import EntityVerificationWorker

__all__ = ['ResearchWorker', 'WebAnglesWorker', 'EntityVerificationWorker']


def start_all_workers():
    """Start all background workers."""
    from .research import start_worker as start_research
    from .web_angles import start_worker as start_angles
    from .entity_verification import start_worker as start_verify

    start_research()
    start_angles()
    start_verify()


def stop_all_workers():
    """Stop all background workers."""
    from .research import stop_worker as stop_research
    from .web_angles import stop_worker as stop_angles
    from .entity_verification import stop_worker as stop_verify

    stop_research()
    stop_angles()
    stop_verify()


def get_all_stats():
    """Get stats from all workers."""
    from .research import get_worker as get_research
    from .web_angles import get_worker as get_angles
    from .entity_verification import get_worker as get_verify

    return {
        "research": {
            "running": get_research().is_running(),
            **get_research().stats.__dict__
        },
        "web_angles": {
            "running": get_angles().is_running(),
            **get_angles().stats.__dict__
        },
        "entity_verification": {
            "running": get_verify().is_running(),
            **get_verify().stats.__dict__
        }
    }
