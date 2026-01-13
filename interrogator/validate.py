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

    Tracks entities, co-occurrences, and correlation strength.
    Correlations are built AND destroyed based on evidence:
    - Co-occurrence strengthens correlation
    - Independent occurrence weakens correlation
    """
    # Raw counts
    entity_counts: Counter = field(default_factory=Counter)
    cooccurrence_counts: Counter = field(default_factory=Counter)  # "A|||B" -> count
    sentence_cooccurrence_counts: Counter = field(default_factory=Counter)  # Same-sentence pairs (stronger)
    by_model: Dict[str, Counter] = field(default_factory=dict)

    # Track when entities last produced new connections (for dead-end detection)
    entity_last_new_connection: Dict[str, int] = field(default_factory=dict)  # entity -> corpus_size when last new connection
    entity_connection_counts: Dict[str, int] = field(default_factory=dict)  # entity -> unique connections count

    # Metadata
    corpus_size: int = 0
    refusal_count: int = 0

    # Thresholds
    entity_threshold: int = 3
    cooccurrence_threshold: int = 2

    # Caches (cleared on add_response)
    _connections_cache: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)
    _scores_cache: Dict[str, float] = field(default_factory=dict)
    _dead_ends_cache: Optional[List[str]] = field(default=None)
    _live_threads_cache: Optional[List[str]] = field(default=None)
    _scored_entities_cache: Optional[List[Tuple[str, float, int]]] = field(default=None)

    def _clear_caches(self):
        """Clear all computed caches."""
        self._connections_cache.clear()
        self._scores_cache.clear()
        self._dead_ends_cache = None
        self._live_threads_cache = None
        self._scored_entities_cache = None

    def add_response(self, entities: List[str], model: str, is_refusal: bool = False,
                     sentence_pairs: Optional[List[Tuple[str, str]]] = None):
        """
        Add entities from a single response with correlation tracking.

        Args:
            entities: All entities from response
            model: Model that generated response
            is_refusal: Whether response was a refusal
            sentence_pairs: Entity pairs from same sentence (stronger correlation signal)
        """
        self.corpus_size += 1
        self._clear_caches()

        if is_refusal:
            self.refusal_count += 1
            return

        # Count entities
        for e in entities:
            self.entity_counts[e] += 1

            if model not in self.by_model:
                self.by_model[model] = Counter()
            self.by_model[model][e] += 1

        # Track sentence-level co-occurrences (strongest signal)
        if sentence_pairs:
            for e1, e2 in sentence_pairs:
                pair = "|||".join(sorted([e1, e2]))
                self.sentence_cooccurrence_counts[pair] += 1

        # Count document-level co-occurrences
        entity_list = list(set(entities))
        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                e1, e2 = entity_list[i], entity_list[j]
                pair = "|||".join(sorted([e1, e2]))

                # Track if this is a NEW connection for each entity
                was_new = self.cooccurrence_counts[pair] == 0

                self.cooccurrence_counts[pair] += 1

                # Update connection tracking for dead-end detection
                if was_new:
                    for e in [e1, e2]:
                        self.entity_last_new_connection[e] = self.corpus_size
                        self.entity_connection_counts[e] = self.entity_connection_counts.get(e, 0) + 1

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
        if entity in self._connections_cache:
            return self._connections_cache[entity]

        connections = []
        for pair, count in self.cooccurrence_counts.items():
            if count >= self.cooccurrence_threshold:
                e1, e2 = pair.split("|||")
                if e1 == entity:
                    connections.append((e2, count))
                elif e2 == entity:
                    connections.append((e1, count))
        result = sorted(connections, key=lambda x: -x[1])
        self._connections_cache[entity] = result
        return result

    def base_score(self, entity: str) -> float:
        """
        Base score without connection quality - just frequency × specificity.
        Used for dead-end detection to avoid recursion.
        """
        frequency = self.entity_counts.get(entity, 0)
        # Estimate specificity from text
        words = entity.split()
        word_count = len(words)
        specificity = 1.0 + (word_count - 1) * 0.5
        from .extract import is_organization, is_year
        if is_organization(entity):
            specificity *= 1.5
        if is_year(entity):
            specificity *= 1.2
        return frequency * specificity

    def connection_quality(self, entity: str) -> float:
        """
        Measure the quality of an entity's connections.

        High quality = connects to high-scoring specific entities (live thread)
        Low quality = connects only to noise/generic entities (dead end)

        Returns average base_score of connected entities, or 0 if no connections.
        """
        connections = self.get_connections(entity)
        if not connections:
            return 0.0

        total_quality = 0.0
        for connected_entity, count in connections:
            # Weight by co-occurrence strength
            total_quality += self.base_score(connected_entity) * count

        total_weight = sum(c for _, c in connections)
        return total_quality / total_weight if total_weight > 0 else 0.0

    def correlation_strength(self, e1: str, e2: str) -> float:
        """
        Compute correlation strength between two entities.

        Uses PMI-like calculation:
        - High co-occurrence relative to independent occurrence = strong
        - Low co-occurrence relative to frequency = weak/destroyed

        Sentence-level co-occurrence weighted 3x stronger than document-level.
        """
        pair = "|||".join(sorted([e1, e2]))
        doc_cooccur = self.cooccurrence_counts.get(pair, 0)
        sent_cooccur = self.sentence_cooccurrence_counts.get(pair, 0)

        # Weighted co-occurrence (sentence = 3x document)
        weighted_cooccur = doc_cooccur + (sent_cooccur * 2)  # sentence already counted in doc

        freq_e1 = self.entity_counts.get(e1, 1)
        freq_e2 = self.entity_counts.get(e2, 1)

        # Expected co-occurrence if independent
        expected = (freq_e1 * freq_e2) / max(self.corpus_size, 1)

        # PMI-like score (actual vs expected)
        if expected > 0:
            return weighted_cooccur / expected
        return weighted_cooccur

    def is_dead_end(self, entity: str, threshold: float = 10.0) -> bool:
        """
        An entity is a dead end if:
        1. It only connects to low-quality/noise entities, OR
        2. It hasn't produced new connections recently

        Dead ends = paths that stop yielding new information.
        """
        connections = self.get_connections(entity)
        if not connections:
            return False  # Can't be dead end without connections yet

        # Check if entity has stopped producing new connections
        last_new = self.entity_last_new_connection.get(entity, 0)
        responses_since_new = self.corpus_size - last_new

        # If no new connections in last 20% of corpus, likely a dead end
        if self.corpus_size > 50 and responses_since_new > self.corpus_size * 0.2:
            return True

        return self.connection_quality(entity) < threshold

    @property
    def dead_ends(self) -> List[str]:
        """Get all entities identified as dead ends."""
        if self._dead_ends_cache is not None:
            return self._dead_ends_cache
        self._dead_ends_cache = [
            e for e in self.validated_entities.keys()
            if self.is_dead_end(e)
        ]
        return self._dead_ends_cache

    @property
    def live_threads(self) -> List[str]:
        """Get entities with high-quality connections (not dead ends)."""
        if self._live_threads_cache is not None:
            return self._live_threads_cache
        self._live_threads_cache = [
            e for e in self.validated_entities.keys()
            if self.get_connections(e) and not self.is_dead_end(e)
        ]
        return self._live_threads_cache

    def score_entity(self, entity: str) -> float:
        """
        Score an entity based on frequency, specificity, connectivity, and connection quality.

        Higher score = more important for narrative.
        Dead ends (leading only to noise) get penalized.
        """
        if entity in self._scores_cache:
            return self._scores_cache[entity]

        frequency = self.entity_counts.get(entity, 0)
        connections = len(self.get_connections(entity))
        base = score_concept(entity, frequency, connections)

        # Apply dead-end penalty - entities that only lead to noise get downweighted
        if self.is_dead_end(entity):
            base *= 0.3  # Significant penalty for dead ends

        self._scores_cache[entity] = base
        return base

    @property
    def scored_entities(self) -> List[Tuple[str, float, int]]:
        """
        All validated entities with scores.

        Returns list of (entity, score, frequency) sorted by score.
        """
        if self._scored_entities_cache is not None:
            return self._scored_entities_cache
        result = []
        for entity, freq in self.validated_entities.items():
            score = self.score_entity(entity)
            result.append((entity, score, freq))
        self._scored_entities_cache = sorted(result, key=lambda x: -x[1])
        return self._scored_entities_cache

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
            # Dead-end detection: entities leading only to public/generic info
            "dead_ends": self.dead_ends[:20],
            "live_threads": self.live_threads[:20],
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
