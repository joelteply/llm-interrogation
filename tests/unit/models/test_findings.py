"""Unit tests for Findings model."""

import pytest
from models import Findings


class TestFindings:
    """Test Findings model."""

    def test_create_empty(self):
        f = Findings()
        assert f.corpus_size == 0
        assert f.refusal_count == 0
        assert f.entity_counts == {}

    def test_add_entity(self):
        f = Findings()
        f.add_entity("John Doe", "gpt-4", corpus_index=0)

        assert f.entity_counts["John Doe"] == 1
        assert "gpt-4" in f.by_model
        assert f.by_model["gpt-4"]["John Doe"] == 1
        assert f.entity_last_activity["John Doe"] == 0

    def test_add_entity_multiple_times(self):
        f = Findings()
        f.add_entity("John Doe", "gpt-4", corpus_index=0)
        f.add_entity("John Doe", "gpt-4", corpus_index=1)
        f.add_entity("John Doe", "claude", corpus_index=2)

        assert f.entity_counts["John Doe"] == 3
        assert f.by_model["gpt-4"]["John Doe"] == 2
        assert f.by_model["claude"]["John Doe"] == 1

    def test_add_cooccurrence(self):
        f = Findings()
        f.add_cooccurrence("John Doe", "Acme Corp")

        # Should be normalized (alphabetical order)
        key = "Acme Corp|||John Doe"
        assert f.cooccurrence_counts[key] == 1

    def test_add_cooccurrence_same_sentence(self):
        f = Findings()
        f.add_cooccurrence("John Doe", "Acme Corp", same_sentence=True)

        key = "Acme Corp|||John Doe"
        assert f.cooccurrence_counts[key] == 1
        assert f.sentence_cooccurrence[key] == 1

    def test_add_relationship(self):
        f = Findings()
        f.add_relationship("John Doe", "works_at", "Acme Corp")

        key = "John Doe|||works_at|||Acme Corp"
        assert f.relationships[key] == 1

    def test_refusal_rate_zero_corpus(self):
        f = Findings()
        assert f.refusal_rate == 0.0

    def test_refusal_rate_calculated(self):
        f = Findings(corpus_size=10, refusal_count=3)
        assert f.refusal_rate == 0.3

    def test_get_scored_entities_filters_by_threshold(self):
        f = Findings(entity_threshold=3)
        f.entity_counts = {"Common": 5, "Rare": 2}
        f.by_model = {"gpt-4": {"Common": 5, "Rare": 2}}

        scored = f.get_scored_entities()
        entities = [s.entity for s in scored]

        assert "Common" in entities
        assert "Rare" not in entities  # Below threshold

    def test_scored_entities_cached(self):
        f = Findings(entity_threshold=1)
        f.entity_counts = {"Test": 5}
        f.by_model = {"gpt-4": {"Test": 5}}

        first = f.get_scored_entities()
        second = f.get_scored_entities()

        assert first is second  # Same object, cached

    def test_cache_invalidated_on_add(self):
        f = Findings(entity_threshold=1)
        f.entity_counts = {"Test": 5}
        f.by_model = {"gpt-4": {"Test": 5}}

        first = f.get_scored_entities()
        f.add_entity("New", "gpt-4", 0)
        second = f.get_scored_entities()

        assert first is not second  # Cache invalidated
