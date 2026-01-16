"""
Project - the root aggregate.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

from .base import BaseEntity
from .corpus import Question


class ProjectSettings(BaseModel):
    """Configurable project settings."""
    runs_per_question: int = 3
    max_tokens: int = 600
    temperature: float = 0.8
    auto_curate: bool = True
    infinite_mode: bool = False


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
