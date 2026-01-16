"""
Entity - a discovered named entity with provenance.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from .base import BaseEntity


class EntityType(str, Enum):
    """Classification of entity types."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    URL = "URL"
    PROJECT = "PROJECT"
    DOCUMENT = "DOCUMENT"
    AMOUNT = "AMOUNT"
    UNKNOWN = "UNKNOWN"


class EntityCategory(str, Enum):
    """Verification status of an entity."""
    UNPROCESSED = "UNPROCESSED"      # Not yet verified
    CORROBORATED = "CORROBORATED"    # Multiple models agree
    UNCORROBORATED = "UNCORROBORATED"  # Single source only
    CONTRADICTED = "CONTRADICTED"    # Models disagree
    PUBLIC = "PUBLIC"                # Found in public sources
    PRIVATE = "PRIVATE"              # Not in public sources


class Provenance(BaseModel):
    """Where an entity was discovered."""
    model: str
    question: str
    response_snippet: str
    timestamp: datetime = Field(default_factory=datetime.now)


class Relationship(BaseModel):
    """A relationship between two entities."""
    target_id: str
    relation_type: str  # "works_at", "emailed", "mentioned_with", etc.
    confidence: float = Field(ge=0.0, le=1.0)
    source: Optional[Provenance] = None


class Entity(BaseEntity):
    """
    A discovered entity with full provenance chain.

    This is THE entity definition. No more scattered dicts.
    """
    id: str
    text: str
    normalized: str  # Lowercase, trimmed canonical form
    entity_type: EntityType = EntityType.UNKNOWN
    category: EntityCategory = EntityCategory.UNPROCESSED

    # Provenance
    discovered_by: Optional[Provenance] = None
    confirmed_by: list[Provenance] = Field(default_factory=list)

    # Scores
    frequency: int = 0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    warmth: float = Field(default=0.0, ge=0.0)  # Connection activity score

    # Relationships
    relationships: list[Relationship] = Field(default_factory=list)

    # Web verification
    web_verified: bool = False
    web_url: Optional[str] = None
    web_snippet: Optional[str] = None

    @classmethod
    def from_text(cls, text: str, **kwargs) -> "Entity":
        """Create entity from raw text."""
        normalized = text.strip().lower()
        entity_id = kwargs.pop("id", normalized.replace(" ", "_"))
        return cls(id=entity_id, text=text.strip(), normalized=normalized, **kwargs)
