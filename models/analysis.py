"""
Analysis models - skeptic feedback, entity verification.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class ConfidenceLevel(str, Enum):
    """Confidence assessment levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class SkepticFeedback(BaseModel):
    """
    Devil's advocate analysis.

    Challenges the working theory with counter-arguments.
    Must back up critiques with its own research.
    """
    model_config = ConfigDict(extra="ignore")

    confirmed_facts: list[str] = Field(default_factory=list)  # Public facts skeptic verified
    weakest_link: Optional[str] = None
    what_i_found: Optional[str] = None  # Skeptic's own research findings
    alternative_explanation: Optional[str] = None
    circular_evidence: list[str] = Field(default_factory=list)
    counter_questions: list[str] = Field(default_factory=list)
    missing_research: Optional[str] = None
    confidence: str = "LOW"  # Flexible - can contain extra text
    updated_at: Optional[datetime] = None  # Set explicitly when created, not on load

    # What theory was being critiqued (for history context)
    theory_snapshot: Optional[str] = None

    # Raw LLM response for debugging
    raw: Optional[str] = None

    def summary_for_rag(self) -> str:
        """Generate a summary for RAG context."""
        parts = []
        if self.confirmed_facts:
            parts.append(f"Skeptic confirmed: {'; '.join(self.confirmed_facts[:5])}")
        if self.what_i_found:
            parts.append(f"Skeptic's research: {self.what_i_found[:200]}")
        if self.weakest_link:
            parts.append(f"Weakest link: {self.weakest_link}")
        if self.alternative_explanation:
            parts.append(f"Alternative: {self.alternative_explanation}")
        if self.counter_questions:
            parts.append(f"Questions: {'; '.join(self.counter_questions[:3])}")
        if self.confidence:
            parts.append(f"Confidence: {self.confidence}")
        return " | ".join(parts) if parts else ""

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Extract confidence level from potentially verbose string."""
        conf_upper = self.confidence.upper()
        if conf_upper.startswith("HIGH"):
            return ConfidenceLevel.HIGH
        if conf_upper.startswith("MEDIUM"):
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    @property
    def has_content(self) -> bool:
        """Check if feedback has any meaningful content."""
        return bool(
            self.weakest_link or
            self.counter_questions or
            self.circular_evidence
        )


class VerifiedEntity(BaseModel):
    """An entity with verification status."""
    entity: str
    url: Optional[str] = None
    snippet: Optional[str] = None
    verified_at: datetime = Field(default_factory=datetime.now)


class EntityVerification(BaseModel):
    """
    Entity verification results.

    Separates PUBLIC (found in web search) from PRIVATE (not found).
    """
    verified: list[VerifiedEntity] = Field(default_factory=list)    # PUBLIC
    unverified: list[VerifiedEntity] = Field(default_factory=list)  # PRIVATE
    unknown: list[str] = Field(default_factory=list)                 # Not yet checked
    summary: str = ""
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def public_entities(self) -> list[str]:
        """Entities found in public sources."""
        return [v.entity for v in self.verified]

    @property
    def private_entities(self) -> list[str]:
        """Entities NOT found in public sources - potentially valuable."""
        return [v.entity for v in self.unverified]

    def mark_verified(self, entity: str, url: str = None, snippet: str = None) -> None:
        """Mark entity as publicly verified."""
        # Remove from unknown/unverified if present
        if entity in self.unknown:
            self.unknown.remove(entity)
        self.unverified = [v for v in self.unverified if v.entity != entity]

        # Add to verified if not already there
        if not any(v.entity == entity for v in self.verified):
            self.verified.append(VerifiedEntity(
                entity=entity,
                url=url,
                snippet=snippet,
            ))
        self.updated_at = datetime.now()

    def mark_unverified(self, entity: str) -> None:
        """Mark entity as not publicly verifiable (private)."""
        if entity in self.unknown:
            self.unknown.remove(entity)

        if not any(v.entity == entity for v in self.unverified):
            self.unverified.append(VerifiedEntity(entity=entity))
        self.updated_at = datetime.now()


class WebLead(BaseModel):
    """A lead discovered from web research."""
    source: str  # "news", "wiki", "court", etc.
    title: str
    url: str
    snippet: str
    relevance: float = Field(ge=0.0, le=1.0)
    discovered_at: datetime = Field(default_factory=datetime.now)


class WebLeads(BaseModel):
    """Accumulated web leads for a project."""
    news_angles: list[WebLead] = Field(default_factory=list)
    wiki_context: list[WebLead] = Field(default_factory=list)
    court_records: list[WebLead] = Field(default_factory=list)
    other: list[WebLead] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_lead(self, lead: WebLead, category: str = "other") -> bool:
        """Add lead if not duplicate."""
        target = getattr(self, category, self.other)
        if any(l.url == lead.url for l in target):
            return False
        target.append(lead)
        self.updated_at = datetime.now()
        return True
