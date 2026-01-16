"""
Repository base classes - define the interface.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, Iterator

from models import (
    Project,
    ProbeResponse,
    Entity,
    Findings,
    ResearchDocument,
    SkepticFeedback,
    EntityVerification,
)

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Abstract base for entity repositories."""

    @abstractmethod
    def get(self, id: str) -> Optional[T]:
        """Get entity by ID."""
        pass

    @abstractmethod
    def save(self, entity: T) -> None:
        """Save entity."""
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete entity by ID. Returns True if deleted."""
        pass

    @abstractmethod
    def list(self) -> list[T]:
        """List all entities."""
        pass

    @abstractmethod
    def exists(self, id: str) -> bool:
        """Check if entity exists."""
        pass


class ProjectRepository(BaseRepository[Project]):
    """Repository for projects."""

    @abstractmethod
    def get_findings(self, project_id: str) -> Optional[Findings]:
        """Get aggregated findings for project."""
        pass

    @abstractmethod
    def save_findings(self, project_id: str, findings: Findings) -> None:
        """Save findings for project."""
        pass

    @abstractmethod
    def get_skeptic_feedback(self, project_id: str) -> Optional[SkepticFeedback]:
        """Get skeptic feedback for project."""
        pass

    @abstractmethod
    def save_skeptic_feedback(self, project_id: str, feedback: SkepticFeedback) -> None:
        """Save skeptic feedback."""
        pass

    @abstractmethod
    def get_entity_verification(self, project_id: str) -> Optional[EntityVerification]:
        """Get entity verification results."""
        pass

    @abstractmethod
    def save_entity_verification(self, project_id: str, verification: EntityVerification) -> None:
        """Save entity verification."""
        pass


class CorpusRepository(BaseRepository[ProbeResponse]):
    """Repository for probe responses (corpus)."""

    @abstractmethod
    def append(self, project_id: str, response: ProbeResponse) -> None:
        """Append response to corpus."""
        pass

    @abstractmethod
    def get_for_project(self, project_id: str) -> list[ProbeResponse]:
        """Get all responses for a project."""
        pass

    @abstractmethod
    def count(self, project_id: str) -> int:
        """Count responses for a project."""
        pass

    @abstractmethod
    def iterate(self, project_id: str) -> Iterator[ProbeResponse]:
        """Iterate responses without loading all into memory."""
        pass


class ResearchRepository(BaseRepository[ResearchDocument]):
    """Repository for research documents."""

    @abstractmethod
    def get_for_project(self, project_id: str) -> list[ResearchDocument]:
        """Get all research documents for a project."""
        pass

    @abstractmethod
    def append(self, project_id: str, doc: ResearchDocument) -> bool:
        """Append document. Returns False if duplicate."""
        pass

    @abstractmethod
    def query(self, project_id: str, terms: list[str], limit: int = 5) -> list[ResearchDocument]:
        """Query documents by relevance to terms."""
        pass


class Repository:
    """
    Aggregate repository - provides access to all entity repositories.

    This is what consumers use. Backend implementations provide
    concrete versions of each sub-repository.
    """

    @property
    @abstractmethod
    def projects(self) -> ProjectRepository:
        """Access project repository."""
        pass

    @property
    @abstractmethod
    def corpus(self) -> CorpusRepository:
        """Access corpus repository."""
        pass

    @property
    @abstractmethod
    def research(self) -> ResearchRepository:
        """Access research repository."""
        pass
