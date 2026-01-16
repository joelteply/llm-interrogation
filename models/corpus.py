"""
Corpus models - questions asked and responses received.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class QuestionTechnique(str, Enum):
    """Interrogation techniques."""
    DIRECT = "direct"
    SCHARFF = "scharff"
    TIMELINE = "timeline"
    RELATIONSHIP = "relationship"
    SPECIFICITY = "specificity"
    PERIPHERAL = "peripheral"
    HYPOTHETICAL = "hypothetical"
    CONTRAST = "contrast"
    SKEPTIC_COUNTER = "skeptic_counter"
    CUSTOM = "custom"


class Question(BaseModel):
    """A generated question for the probe."""
    model_config = ConfigDict(extra="allow")

    question: str = ""  # The question text
    technique: str = "direct"  # Interrogation technique used
    template: Optional[str] = None  # Template that generated this
    target_entity: Optional[str] = None  # Entity this targets
    color: Optional[str] = None  # UI display color

    @property
    def text(self) -> str:
        """Alias for compatibility."""
        return self.question


class ProbeResponse(BaseModel):
    """
    A single response from a model.

    This is the atomic unit of the corpus.
    """
    model_config = ConfigDict(extra="ignore")  # Ignore unknown fields from legacy data

    # Identity
    question_index: int
    run_index: int = 0  # Default for legacy data
    model: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Content
    question: str
    response: str

    # Extraction results
    entities: list[str] = Field(default_factory=list)
    discovered_entities: list[str] = Field(default_factory=list)  # Genuine findings
    introduced_entities: list[str] = Field(default_factory=list)  # Echoed from query

    # Classification
    is_refusal: bool = False
    technique: Optional[str] = None  # Flexible - any technique string (legacy data has non-enum values)

    @property
    def prompt(self) -> str:
        """Alias for compatibility."""
        return self.question

    @property
    def text(self) -> str:
        """Alias for compatibility."""
        return self.response
