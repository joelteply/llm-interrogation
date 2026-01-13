"""
Statistical validation of extracted entities and co-occurrences.

Uses frequency thresholds to separate signal from noise:
- Entities appearing N+ times = validated
- Pairs appearing together N+ times = validated relationship
"""

from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import Counter
from .extract import score_concept


@dataclass
class Findings:
    """
    Accumulated findings from probing.

    Tracks entities, co-occurrences, and computed scores.
    """
    # Raw counts
    entity_counts: Counter = field(default_factory=Counter)
    cooccurrence_counts: Counter = field(default_factory=Counter)  # "A|||B" -> count
    by_model: Dict[str, Counter] = field(default_factory=dict)

    # Metadata
    corpus_size: int = 0
    refusal_count: int = 0

    # Thresholds
    entity_threshold: int = 3
    cooccurrence_threshold: int = 2

    def add_response(self, entities: List[str], model: str, is_refusal: bool = False):
        """Add entities from a single response."""
        self.corpus_size += 1

        if is_refusal:
            self.refusal_count += 1
            return

        # Count entities
        for e in entities:
            self.entity_counts[e] += 1

            if model not in self.by_model:
                self.by_model[model] = Counter()
            self.by_model[model][e] += 1

        # Count co-occurrences (pairs in same response)
        entity_list = list(set(entities))  # Dedupe within response
        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                pair = "|||".join(sorted([entity_list[i], entity_list[j]]))
                self.cooccurrence_counts[pair] += 1

    @property
    def validated_entities(self) -> Dict[str, int]:
        """Entities that pass the frequency threshold."""
        return {
            e: c for e, c in self.entity_counts.items()
            if c >= self.entity_threshold
        }

    @property
    def noise_entities(self) -> Dict[str, int]:
        """Entities below threshold (likely noise/hallucination)."""
        return {
            e: c for e, c in self.entity_counts.items()
            if c < self.entity_threshold
        }

    @property
    def validated_cooccurrences(self) -> List[Tuple[str, str, int]]:
        """
        Co-occurrences that pass the frequency threshold.

        Returns list of (entity1, entity2, count) tuples.
        """
        result = []
        for pair, count in self.cooccurrence_counts.items():
            if count >= self.cooccurrence_threshold:
                e1, e2 = pair.split("|||")
                result.append((e1, e2, count))
        return sorted(result, key=lambda x: -x[2])

    def get_connections(self, entity: str) -> List[Tuple[str, int]]:
        """Get all validated connections for an entity."""
        connections = []
        for pair, count in self.cooccurrence_counts.items():
            if count >= self.cooccurrence_threshold:
                e1, e2 = pair.split("|||")
                if e1 == entity:
                    connections.append((e2, count))
                elif e2 == entity:
                    connections.append((e1, count))
        return sorted(connections, key=lambda x: -x[1])

    def score_entity(self, entity: str) -> float:
        """
        Score an entity based on frequency, specificity, and connectivity.

        Higher score = more important for narrative.
        """
        frequency = self.entity_counts.get(entity, 0)
        connections = len(self.get_connections(entity))
        return score_concept(entity, frequency, connections)

    @property
    def scored_entities(self) -> List[Tuple[str, float, int]]:
        """
        All validated entities with scores.

        Returns list of (entity, score, frequency) sorted by score.
        """
        result = []
        for entity, freq in self.validated_entities.items():
            score = self.score_entity(entity)
            result.append((entity, score, freq))
        return sorted(result, key=lambda x: -x[1])

    @property
    def refusal_rate(self) -> float:
        """Fraction of responses that were refusals."""
        if self.corpus_size == 0:
            return 0
        return self.refusal_count / self.corpus_size

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "entities": dict(self.entity_counts.most_common(50)),
            "cooccurrences": [
                {"entities": [e1, e2], "count": c}
                for e1, e2, c in self.validated_cooccurrences[:100]
            ],
            "scored_entities": [
                {"entity": e, "score": round(s, 2), "frequency": f}
                for e, s, f in self.scored_entities[:30]
            ],
            "by_model": {
                m: dict(c.most_common(20))
                for m, c in self.by_model.items()
            },
            "corpus_size": self.corpus_size,
            "refusal_rate": round(self.refusal_rate, 3),
            "validated_count": len(self.validated_entities),
            "noise_count": len(self.noise_entities),
        }


def validate_entities(
    entity_counts: Counter,
    threshold: int = 3
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Split entities into validated (signal) and noise.

    Args:
        entity_counts: Counter of entity -> frequency
        threshold: Minimum frequency to be considered validated

    Returns:
        (validated, noise) dicts
    """
    validated = {e: c for e, c in entity_counts.items() if c >= threshold}
    noise = {e: c for e, c in entity_counts.items() if c < threshold}
    return validated, noise


def validate_cooccurrences(
    cooccurrence_counts: Counter,
    threshold: int = 2
) -> List[Tuple[str, str, int]]:
    """
    Filter co-occurrences by frequency threshold.

    Args:
        cooccurrence_counts: Counter of "A|||B" -> frequency
        threshold: Minimum frequency to be considered validated

    Returns:
        List of (entity1, entity2, count) tuples, sorted by count desc
    """
    result = []
    for pair, count in cooccurrence_counts.items():
        if count >= threshold:
            e1, e2 = pair.split("|||")
            result.append((e1, e2, count))
    return sorted(result, key=lambda x: -x[2])
