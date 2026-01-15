"""
Research Adapter Base - Clean interface for all research sources.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ResearchDoc:
    """A document from any research source."""
    id: str
    source: str
    title: str
    url: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class AdapterConfig:
    """Configuration for an adapter."""
    timeout: int = 30
    max_content_size: int = 50000
    env_vars: dict = field(default_factory=dict)  # Required env vars


class ResearchAdapter(ABC):
    """
    Base class for research adapters.

    Each adapter must:
    1. Define name, description
    2. Implement available() - check deps + auth
    3. Implement search() - find documents
    4. Optionally implement fetch() - get single doc by ID
    """

    name: str
    description: str = ""
    config: AdapterConfig = field(default_factory=AdapterConfig)

    def __init__(self):
        self.config = self._default_config()

    def _default_config(self) -> AdapterConfig:
        """Override to set adapter-specific config."""
        return AdapterConfig()

    def get_env(self, key: str, default: str = None) -> Optional[str]:
        """Get environment variable."""
        return os.environ.get(key, default)

    def require_env(self, key: str) -> Optional[str]:
        """Get required env var, return None if missing."""
        val = os.environ.get(key)
        if not val:
            print(f"[{self.name}] Missing required env var: {key}")
        return val

    @abstractmethod
    def available(self) -> bool:
        """Check if adapter can be used (deps, auth, etc)."""
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[ResearchDoc]:
        """Search for documents. Returns list of ResearchDoc."""
        pass

    def fetch(self, doc_id: str) -> Optional[ResearchDoc]:
        """Fetch single document by ID. Override if supported."""
        return None
