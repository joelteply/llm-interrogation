"""
Findings - aggregated extraction results.

Single definition replacing the three scattered implementations.
"""

from typing import Optional
from pydantic import BaseModel, Field
from collections import Counter


class EntityScore(BaseModel):
    """An entity with its computed scores."""
    entity: str
    frequency: int = 0
    score: float = 0.0
    warmth: float = 0.0  # Recent connection activity
    model_count: int = 0  # How many models mentioned it
    is_dead_end: bool = False
    is_live_thread: bool = False


class Cooccurrence(BaseModel):
    """Two entities appearing together."""
    entity_a: str
    entity_b: str
    count: int = 0
    sentence_count: int = 0  # Stronger signal - same sentence


class RelationshipExtract(BaseModel):
    """An extracted relationship between entities."""
    subject: str
    predicate: str
    obj: str  # 'object' is reserved
    count: int = 0
    confidence: float = 0.0


class Findings(BaseModel):
    """
    Aggregated findings from probing.

    This replaces:
    - interrogator/validate.py:Findings
    - routes/analyze/findings.py:Finding
    - Various dict structures

    Single source of truth for extraction results.
    """
    # Entity counts
    entity_counts: dict[str, int] = Field(default_factory=dict)
    by_model: dict[str, dict[str, int]] = Field(default_factory=dict)

    # Cooccurrence
    cooccurrence_counts: dict[str, int] = Field(default_factory=dict)  # "A|||B" -> count
    sentence_cooccurrence: dict[str, int] = Field(default_factory=dict)

    # Relationships
    relationships: dict[str, int] = Field(default_factory=dict)  # "A|||pred|||B" -> count

    # Tracking
    first_mentions: dict[str, list[str]] = Field(default_factory=dict)  # model -> entities
    entity_last_activity: dict[str, int] = Field(default_factory=dict)  # entity -> corpus_index

    # Stats
    corpus_size: int = 0
    refusal_count: int = 0

    # Thresholds
    entity_threshold: int = 3
    cooccurrence_threshold: int = 2

    # Computed (cached)
    _scored_entities: Optional[list[EntityScore]] = None
    _dead_ends: Optional[set[str]] = None
    _live_threads: Optional[set[str]] = None

    def add_entity(self, entity: str, model: str, corpus_index: int) -> None:
        """Record an entity mention."""
        self.entity_counts[entity] = self.entity_counts.get(entity, 0) + 1

        if model not in self.by_model:
            self.by_model[model] = {}
        self.by_model[model][entity] = self.by_model[model].get(entity, 0) + 1

        self.entity_last_activity[entity] = corpus_index
        self._invalidate_cache()

    def add_cooccurrence(self, entity_a: str, entity_b: str, same_sentence: bool = False) -> None:
        """Record two entities appearing together."""
        key = f"{min(entity_a, entity_b)}|||{max(entity_a, entity_b)}"
        self.cooccurrence_counts[key] = self.cooccurrence_counts.get(key, 0) + 1

        if same_sentence:
            self.sentence_cooccurrence[key] = self.sentence_cooccurrence.get(key, 0) + 1

        self._invalidate_cache()

    def add_relationship(self, subject: str, predicate: str, obj: str) -> None:
        """Record a relationship."""
        key = f"{subject}|||{predicate}|||{obj}"
        self.relationships[key] = self.relationships.get(key, 0) + 1
        self._invalidate_cache()

    def get_scored_entities(self) -> list[EntityScore]:
        """Get entities with computed scores."""
        if self._scored_entities is not None:
            return self._scored_entities

        scores = []
        for entity, freq in self.entity_counts.items():
            if freq < self.entity_threshold:
                continue

            model_count = sum(
                1 for m in self.by_model.values()
                if entity in m
            )

            # Warmth: recent activity indicator
            last_seen = self.entity_last_activity.get(entity, 0)
            recency = max(0, self.corpus_size - last_seen)
            warmth = 1.0 / (1 + recency * 0.1) if self.corpus_size > 0 else 0

            # Score: frequency * model diversity * warmth
            score = freq * (1 + model_count * 0.5) * (1 + warmth)

            scores.append(EntityScore(
                entity=entity,
                frequency=freq,
                score=score,
                warmth=warmth,
                model_count=model_count,
            ))

        scores.sort(key=lambda x: x.score, reverse=True)
        self._scored_entities = scores
        return scores

    @property
    def refusal_rate(self) -> float:
        """Percentage of refusals."""
        if self.corpus_size == 0:
            return 0.0
        return self.refusal_count / self.corpus_size

    def _invalidate_cache(self) -> None:
        """Clear computed caches."""
        self._scored_entities = None
        self._dead_ends = None
        self._live_threads = None
