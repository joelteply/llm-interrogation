"""
Domain models - single source of truth for all entities.

Design principles:
- Every entity defined once
- Proper inheritance for shared fields
- Validation at the boundary
- Backend-agnostic (repository handles persistence)
"""

from .base import BaseEntity, TimestampMixin
from .project import Project, ProjectSettings, SeedSource, ProbeTarget, SeedExplorationState
from .corpus import ProbeResponse, Question, QuestionTechnique, Asset
from .entity import Entity, EntityType, EntityCategory, Relationship
from .findings import Findings, EntityScore, Cooccurrence
from .research import ResearchDocument, ResearchQuery
from .analysis import SkepticFeedback, EntityVerification, VerifiedEntity
from .worker import WorkerStats

__all__ = [
    # Base
    "BaseEntity",
    "TimestampMixin",
    # Project
    "Project",
    "ProjectSettings",
    "SeedSource",
    "ProbeTarget",
    "SeedExplorationState",
    # Corpus
    "ProbeResponse",
    "Question",
    "QuestionTechnique",
    "Asset",
    # Entity
    "Entity",
    "EntityType",
    "EntityCategory",
    "Relationship",
    # Findings
    "Findings",
    "EntityScore",
    "Cooccurrence",
    # Research
    "ResearchDocument",
    "ResearchQuery",
    # Analysis
    "SkepticFeedback",
    "EntityVerification",
    "VerifiedEntity",
    # Worker
    "WorkerStats",
]
