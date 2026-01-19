"""
Project - the root aggregate.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

from .base import BaseEntity
from .corpus import Question, Asset


class ProjectSettings(BaseModel):
    """Configurable project settings."""
    runs_per_question: int = 3
    max_tokens: int = 600
    temperature: float = 0.8
    auto_curate: bool = True
    infinite_mode: bool = False


class SeedSource(BaseModel):
    """Source material for investigation - path, URL, or raw content."""
    type: str = "none"  # 'path' | 'url' | 'content' | 'none'
    value: str = ""     # The path, URL, or raw content
    content_type: str = "auto"  # 'auto' | 'code' | 'legal' | 'research' | 'general'


class ProbeTarget(BaseModel):
    """An identifier extracted from seed to probe for."""
    identifier: str           # The thing to probe for
    source_file: str = ""     # Where it came from
    context: str = ""         # Surrounding context
    probed: bool = False      # Has it been probed?
    hit: bool = False         # Did LLMs recognize it?
    hit_count: int = 0        # How many models recognized it
    embedding: list[float] = Field(default_factory=list)  # Semantic embedding vector


class SeedExplorationState(BaseModel):
    """Tracks progress exploring seed content."""
    explored_paths: list[str] = Field(default_factory=list)  # Already extracted from
    probe_queue: list[ProbeTarget] = Field(default_factory=list)  # Waiting to probe
    probed: list[ProbeTarget] = Field(default_factory=list)  # Already checked
    hot_zones: list[str] = Field(default_factory=list)  # Areas with hits - prioritize
    chunked_files: list[str] = Field(default_factory=list)  # Files already semantically chunked


class Project(BaseEntity):
    """
    A research project.

    This is THE project definition. project.json maps directly to this.
    """
    model_config = ConfigDict(extra="allow")  # Allow legacy fields during migration

    # Identity
    name: str
    topic: str = ""

    # Model selection
    selected_models: list[str] = Field(default_factory=list)

    # Entity curation
    hidden_entities: list[str] = Field(default_factory=list)   # Excluded
    promoted_entities: list[str] = Field(default_factory=list)  # Focused

    # Investigation angles
    angles: list[str] = Field(default_factory=list)

    # Working state
    narrative: str = ""
    narrative_updated: Optional[datetime] = None
    user_notes: str = ""

    # Question queue
    questions: list[Question] = Field(default_factory=list)

    # Settings
    settings: ProjectSettings = Field(default_factory=ProjectSettings)

    # Research context (accumulated from workers)
    research_context: str = ""
    research_updated: Optional[datetime] = None

    # Assets - curated evidence pointers
    assets: list[Asset] = Field(default_factory=list)

    # Seed source - private data to probe for
    seed: SeedSource = Field(default_factory=SeedSource)
    seed_state: SeedExplorationState = Field(default_factory=SeedExplorationState)

    # Investigation goal (auto-detected or user-specified)
    goal: str = ""  # e.g., "find_leaks", "competitive_intel", "research", "general"

    def hide_entity(self, entity: str) -> None:
        """Add entity to exclusion list."""
        normalized = entity.strip()
        if normalized and normalized not in self.hidden_entities:
            self.hidden_entities.append(normalized)
            # Remove from promoted if present
            if normalized in self.promoted_entities:
                self.promoted_entities.remove(normalized)
            self.touch()

    def promote_entity(self, entity: str) -> None:
        """Add entity to focus list."""
        normalized = entity.strip()
        if normalized and normalized not in self.promoted_entities:
            self.promoted_entities.append(normalized)
            # Remove from hidden if present
            if normalized in self.hidden_entities:
                self.hidden_entities.remove(normalized)
            self.touch()

    def add_angle(self, angle: str) -> None:
        """Add investigation angle."""
        normalized = angle.strip()
        if normalized and normalized not in self.angles:
            self.angles.append(normalized)
            self.touch()

    def update_narrative(self, narrative: str) -> None:
        """Update the working theory."""
        self.narrative = narrative
        self.narrative_updated = datetime.now()
        self.touch()
