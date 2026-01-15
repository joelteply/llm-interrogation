"""
Research Adapter Base Class.

Each source (DocumentCloud, PACER, etc.) implements this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ResearchDoc:
    """A document from any research source."""
    id: str
    source: str  # adapter name
    title: str
    url: str
    content: str
    metadata: dict = field(default_factory=dict)


class ResearchAdapter(ABC):
    """Base class for research source adapters."""

    name: str  # 'documentcloud', 'pacer', 'web', etc.
    description: str = ""

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[ResearchDoc]:
        """Search this source. Returns list of documents."""
        pass

    @abstractmethod
    def available(self) -> bool:
        """Check if this source is available (deps installed, API keys set, etc)."""
        pass

    def fetch(self, doc_id: str) -> Optional[ResearchDoc]:
        """Fetch a specific document by ID. Optional to implement."""
        return None
