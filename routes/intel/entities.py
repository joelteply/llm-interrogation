"""
Entity data structures and storage.

Tracks entities with full provenance and confidence scoring.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, List
from datetime import datetime
import json
import hashlib
from pathlib import Path


@dataclass
class ResponseRef:
    """Reference to a response that mentioned an entity."""
    model: str
    response_id: str
    question_id: str
    question_text: str
    response_text: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self):
        return {
            "model": self.model,
            "response_id": self.response_id,
            "question_id": self.question_id,
            "question_text": self.question_text[:200],
            "response_text": self.response_text[:500],
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class WebResult:
    """Result from web search for an entity."""
    query: str
    url: str
    title: str
    snippet: str
    matches_context: bool  # Does this match the specific claim context?

    def to_dict(self):
        return {
            "query": self.query,
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet[:300],
            "matches_context": self.matches_context
        }


@dataclass
class DrillScores:
    """Results from the drill protocol."""
    consistency: float = 0.0  # 0-1: stable under repetition
    contradiction: float = 0.0  # 0-1: resists false alternatives
    detail: float = 0.0  # 0-1: natural detail falloff
    provenance: float = 0.0  # 0-1: can explain how it knows
    peripheral: float = 0.0  # 0-1: depth in unexpected directions
    control: float = 0.0  # 0-1: accurate on known facts

    @property
    def total(self) -> float:
        """Total score out of 60."""
        return sum([
            self.consistency,
            self.contradiction,
            self.detail,
            self.provenance,
            self.peripheral,
            self.control
        ]) * 10

    @property
    def average(self) -> float:
        """Average score 0-1."""
        scores = [
            self.consistency,
            self.contradiction,
            self.detail,
            self.provenance,
            self.peripheral,
            self.control
        ]
        return sum(scores) / len(scores) if scores else 0

    def to_dict(self):
        return {
            "consistency": self.consistency,
            "contradiction": self.contradiction,
            "detail": self.detail,
            "provenance": self.provenance,
            "peripheral": self.peripheral,
            "control": self.control,
            "total": self.total,
            "average": self.average
        }


@dataclass
class Relationship:
    """Relationship between two entities."""
    target_entity_id: str
    relation_type: str  # "mentioned_with", "works_at", "emailed", etc.
    confidence: float
    source_response_id: str

    def to_dict(self):
        return {
            "target_entity_id": self.target_entity_id,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "source_response_id": self.source_response_id
        }


Category = Literal["CORROBORATED", "UNCORROBORATED", "CONTRADICTED", "UNPROCESSED"]


@dataclass
class Entity:
    """
    An entity extracted from model responses.

    Tracks full provenance, categorization, drill results, and confidence.
    """
    id: str
    text: str
    entity_type: str = "UNKNOWN"  # PERSON, ORG, EMAIL, PROJECT, DATE, etc.

    # Category
    category: Category = "UNPROCESSED"

    # Provenance - who said it first, who confirmed
    originated_from: Optional[ResponseRef] = None
    confirmed_by: List[ResponseRef] = field(default_factory=list)

    # Analysis
    drill_scores: Optional[DrillScores] = None
    confidence: float = 0.0
    web_results: List[WebResult] = field(default_factory=list)

    # Graph
    relationships: List[Relationship] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def source_models(self) -> set:
        """All unique models that mentioned this entity."""
        models = set()
        if self.originated_from:
            models.add(self.originated_from.model)
        for ref in self.confirmed_by:
            models.add(ref.model)
        return models

    @property
    def independent_sources(self) -> int:
        """Count of independent model sources (not same model echoing)."""
        return len(self.source_models)

    @property
    def total_mentions(self) -> int:
        """Total times mentioned (including echoes)."""
        count = 1 if self.originated_from else 0
        count += len(self.confirmed_by)
        return count

    @property
    def is_single_source(self) -> bool:
        """True if only one model has mentioned this."""
        return self.independent_sources <= 1

    @property
    def is_echo_chamber(self) -> bool:
        """True if multiple mentions but all from same model."""
        return self.total_mentions > 1 and self.is_single_source

    def add_confirmation(self, ref: ResponseRef):
        """Add a confirmation reference."""
        self.confirmed_by.append(ref)
        self.updated_at = datetime.now()

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "entity_type": self.entity_type,
            "category": self.category,
            "originated_from": self.originated_from.to_dict() if self.originated_from else None,
            "confirmed_by": [ref.to_dict() for ref in self.confirmed_by],
            "drill_scores": self.drill_scores.to_dict() if self.drill_scores else None,
            "confidence": self.confidence,
            "web_results": [wr.to_dict() for wr in self.web_results],
            "relationships": [rel.to_dict() for rel in self.relationships],
            "source_models": list(self.source_models),
            "independent_sources": self.independent_sources,
            "total_mentions": self.total_mentions,
            "is_single_source": self.is_single_source,
            "is_echo_chamber": self.is_echo_chamber,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


def make_entity_id(text: str) -> str:
    """Generate consistent ID for an entity."""
    normalized = text.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


class EntityStore:
    """
    Storage for entities with provenance tracking.
    """

    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)
        self.entities_file = self.project_dir / "entities.json"
        self.entities: Dict[str, Entity] = {}
        self._load()

    def _load(self):
        """Load entities from disk."""
        if self.entities_file.exists():
            try:
                with open(self.entities_file) as f:
                    data = json.load(f)
                    for eid, edata in data.items():
                        self.entities[eid] = self._entity_from_dict(edata)
            except Exception as e:
                print(f"[EntityStore] Failed to load: {e}")

    def _entity_from_dict(self, data: dict) -> Entity:
        """Reconstruct Entity from dict."""
        entity = Entity(
            id=data["id"],
            text=data["text"],
            entity_type=data.get("entity_type", "UNKNOWN"),
            category=data.get("category", "UNPROCESSED"),
            confidence=data.get("confidence", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now()
        )

        if data.get("originated_from"):
            orig = data["originated_from"]
            entity.originated_from = ResponseRef(
                model=orig["model"],
                response_id=orig["response_id"],
                question_id=orig["question_id"],
                question_text=orig.get("question_text", ""),
                response_text=orig.get("response_text", ""),
                timestamp=datetime.fromisoformat(orig["timestamp"]) if "timestamp" in orig else datetime.now()
            )

        for conf in data.get("confirmed_by", []):
            entity.confirmed_by.append(ResponseRef(
                model=conf["model"],
                response_id=conf["response_id"],
                question_id=conf["question_id"],
                question_text=conf.get("question_text", ""),
                response_text=conf.get("response_text", ""),
                timestamp=datetime.fromisoformat(conf["timestamp"]) if "timestamp" in conf else datetime.now()
            ))

        if data.get("drill_scores"):
            ds = data["drill_scores"]
            entity.drill_scores = DrillScores(
                consistency=ds.get("consistency", 0),
                contradiction=ds.get("contradiction", 0),
                detail=ds.get("detail", 0),
                provenance=ds.get("provenance", 0),
                peripheral=ds.get("peripheral", 0),
                control=ds.get("control", 0)
            )

        for wr in data.get("web_results", []):
            entity.web_results.append(WebResult(
                query=wr["query"],
                url=wr["url"],
                title=wr["title"],
                snippet=wr["snippet"],
                matches_context=wr.get("matches_context", False)
            ))

        for rel in data.get("relationships", []):
            entity.relationships.append(Relationship(
                target_entity_id=rel["target_entity_id"],
                relation_type=rel["relation_type"],
                confidence=rel["confidence"],
                source_response_id=rel["source_response_id"]
            ))

        return entity

    def save(self):
        """Save entities to disk."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        data = {eid: e.to_dict() for eid, e in self.entities.items()}
        with open(self.entities_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_entity(
        self,
        text: str,
        model: str,
        response_id: str,
        question_id: str,
        question_text: str,
        response_text: str,
        entity_type: str = "UNKNOWN"
    ) -> Entity:
        """
        Add an entity or record a confirmation if it already exists.

        Returns the Entity (new or existing).
        """
        eid = make_entity_id(text)
        ref = ResponseRef(
            model=model,
            response_id=response_id,
            question_id=question_id,
            question_text=question_text,
            response_text=response_text
        )

        if eid in self.entities:
            # Existing entity - add confirmation
            self.entities[eid].add_confirmation(ref)
        else:
            # New entity - set as origin
            entity = Entity(
                id=eid,
                text=text,
                entity_type=entity_type,
                originated_from=ref
            )
            self.entities[eid] = entity

        return self.entities[eid]

    def get(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    def get_by_text(self, text: str) -> Optional[Entity]:
        """Get entity by text."""
        eid = make_entity_id(text)
        return self.entities.get(eid)

    def all(self) -> List[Entity]:
        """Get all entities."""
        return list(self.entities.values())

    def uncorroborated(self) -> List[Entity]:
        """Get entities needing drill."""
        return [e for e in self.entities.values() if e.category == "UNCORROBORATED"]

    def by_confidence(self, min_confidence: float = 0.0) -> List[Entity]:
        """Get entities sorted by confidence."""
        filtered = [e for e in self.entities.values() if e.confidence >= min_confidence]
        return sorted(filtered, key=lambda e: -e.confidence)

    def echo_chambers(self) -> List[Entity]:
        """Get entities that are echo chambers (same model, multiple mentions)."""
        return [e for e in self.entities.values() if e.is_echo_chamber]
