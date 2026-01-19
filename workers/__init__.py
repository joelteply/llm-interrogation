"""
Background workers - continuous processing.

All workers extend BaseWorker which handles:
- Thread lifecycle
- Stats tracking via shared WorkerStats model
- Project focus
- Callbacks

Workers:
- InterrogatorWorker: Synthesizes findings into working theory
- ResearchWorker: LLM-directed document research
- WebAnglesWorker: Web angle discovery
- EntityVerificationWorker: PUBLIC/PRIVATE classification
- SkepticWorker: Devil's advocate analysis (challenges the interrogator)
- SeedExplorerWorker: Progressive extraction from seed source material
- SemanticIndexerWorker: Generates embeddings for probe targets
- SemanticChunkerWorker: Chunks seed content into semantic units
"""

from .base import BaseWorker
from .interrogator import InterrogatorWorker
from .research import ResearchWorker
from .web_angles import WebAnglesWorker
from .entity_verification import EntityVerificationWorker
from .skeptic import SkepticWorker
from .seed_explorer import SeedExplorerWorker
from .semantic_indexer import SemanticIndexerWorker, SemanticChunkerWorker

__all__ = [
    'BaseWorker',
    'InterrogatorWorker',
    'ResearchWorker',
    'WebAnglesWorker',
    'EntityVerificationWorker',
    'SkepticWorker',
    'SeedExplorerWorker',
    'SemanticIndexerWorker',
    'SemanticChunkerWorker',
]


def start_all_workers():
    """Start all background workers."""
    from .interrogator import start_worker as start_interrogator
    from .research import start_worker as start_research
    from .web_angles import start_worker as start_angles
    from .entity_verification import start_worker as start_verify
    from .skeptic import start_worker as start_skeptic
    from .seed_explorer import start_worker as start_seed_explorer
    from .semantic_indexer import start_workers as start_semantic

    start_interrogator()
    start_research()
    start_angles()
    start_verify()
    start_skeptic()
    start_seed_explorer()
    start_semantic()


def stop_all_workers():
    """Stop all background workers."""
    from .interrogator import stop_worker as stop_interrogator
    from .research import stop_worker as stop_research
    from .web_angles import stop_worker as stop_angles
    from .entity_verification import stop_worker as stop_verify
    from .skeptic import stop_worker as stop_skeptic
    from .seed_explorer import stop_worker as stop_seed_explorer
    from .semantic_indexer import stop_workers as stop_semantic

    stop_interrogator()
    stop_research()
    stop_angles()
    stop_verify()
    stop_skeptic()
    stop_seed_explorer()
    stop_semantic()


def get_all_stats() -> dict:
    """Get stats from all workers using unified format."""
    from .interrogator import get_worker as get_interrogator
    from .research import get_worker as get_research
    from .web_angles import get_worker as get_angles
    from .entity_verification import get_worker as get_verify
    from .skeptic import get_worker as get_skeptic
    from .seed_explorer import get_worker as get_seed_explorer
    from .semantic_indexer import get_indexer, get_chunker

    return {
        "interrogator": get_interrogator().get_stats(),
        "research": get_research().get_stats(),
        "web_angles": get_angles().get_stats(),
        "entity_verification": get_verify().get_stats(),
        "skeptic": get_skeptic().get_stats(),
        "seed_explorer": get_seed_explorer().get_stats(),
        "semantic_indexer": get_indexer().get_stats(),
        "semantic_chunker": get_chunker().get_stats(),
    }


def set_all_project(project_name: str):
    """Set active project for all workers."""
    from .interrogator import get_worker as get_interrogator
    from .research import get_worker as get_research
    from .skeptic import get_worker as get_skeptic
    from .entity_verification import get_worker as get_verify
    from .web_angles import get_worker as get_angles
    from .seed_explorer import get_worker as get_seed_explorer
    from .semantic_indexer import get_indexer, get_chunker

    get_interrogator().set_project(project_name)
    get_research().set_project(project_name)
    get_skeptic().set_project(project_name)
    get_verify().set_project(project_name)
    get_angles().set_project(project_name)
    get_seed_explorer().set_project(project_name)
    get_indexer().set_project(project_name)
    get_chunker().set_project(project_name)
