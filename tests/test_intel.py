"""
Tests for the Intelligence Extraction System.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from routes.intel.entities import (
    Entity, EntityStore, ResponseRef, DrillScores,
    make_entity_id, WebResult, Relationship
)
from routes.intel.confidence import (
    calculate_confidence, confidence_label, should_drill, prioritize_for_drill
)
from routes.intel.drill import (
    extract_key_facts, calculate_consistency,
    check_contradiction_resistance, check_detail_coherence, check_provenance
)
from routes.intel.graph import EntityGraph


class TestEntityStore:
    """Test entity storage and provenance tracking."""

    def setup_method(self):
        """Create temp directory for tests."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.store = EntityStore(self.temp_dir)

    def teardown_method(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir)

    def test_add_new_entity(self):
        """Adding a new entity sets it as originated_from."""
        entity = self.store.add_entity(
            text="Project Aurora",
            model="gpt-4",
            response_id="r1",
            question_id="q1",
            question_text="What projects exist?",
            response_text="Project Aurora is a data visualization tool."
        )

        assert entity.text == "Project Aurora"
        assert entity.originated_from is not None
        assert entity.originated_from.model == "gpt-4"
        assert len(entity.confirmed_by) == 0

    def test_add_confirmation(self):
        """Adding same entity again adds to confirmed_by."""
        # First mention
        self.store.add_entity(
            text="Project Aurora",
            model="gpt-4",
            response_id="r1",
            question_id="q1",
            question_text="What projects?",
            response_text="Project Aurora"
        )

        # Second mention (different model)
        entity = self.store.add_entity(
            text="Project Aurora",
            model="claude-3",
            response_id="r2",
            question_id="q2",
            question_text="Tell me about projects",
            response_text="Project Aurora is..."
        )

        assert entity.originated_from.model == "gpt-4"
        assert len(entity.confirmed_by) == 1
        assert entity.confirmed_by[0].model == "claude-3"

    def test_independent_sources(self):
        """Count independent model sources correctly."""
        # Add from model A
        self.store.add_entity("Test Entity", "model-a", "r1", "q1", "q", "r")
        # Add from model A again (echo)
        self.store.add_entity("Test Entity", "model-a", "r2", "q2", "q", "r")
        # Add from model B
        entity = self.store.add_entity("Test Entity", "model-b", "r3", "q3", "q", "r")

        assert entity.total_mentions == 3
        assert entity.independent_sources == 2
        assert not entity.is_single_source

    def test_echo_chamber_detection(self):
        """Detect when same model echoes itself."""
        # Add from model A multiple times
        self.store.add_entity("Echo Entity", "model-a", "r1", "q1", "q", "r")
        self.store.add_entity("Echo Entity", "model-a", "r2", "q2", "q", "r")
        entity = self.store.add_entity("Echo Entity", "model-a", "r3", "q3", "q", "r")

        assert entity.total_mentions == 3
        assert entity.independent_sources == 1
        assert entity.is_echo_chamber

    def test_save_and_load(self):
        """Test persistence."""
        self.store.add_entity("Persistent", "model", "r1", "q1", "q", "r")
        self.store.save()

        # Load in new store
        new_store = EntityStore(self.temp_dir)
        entity = new_store.get_by_text("Persistent")

        assert entity is not None
        assert entity.text == "Persistent"


class TestConfidence:
    """Test confidence scoring."""

    def test_base_confidence(self):
        """Base confidence for new entity."""
        entity = Entity(
            id="test",
            text="Test",
            category="UNPROCESSED"
        )
        entity.originated_from = ResponseRef(
            model="gpt-4", response_id="r1", question_id="q1",
            question_text="q", response_text="r"
        )

        conf = calculate_confidence(entity)
        assert 0 < conf < 1

    def test_corroborated_bonus(self):
        """Corroborated entities get bonus."""
        entity = Entity(id="test", text="Test", category="CORROBORATED")
        entity.originated_from = ResponseRef("gpt-4", "r1", "q1", "q", "r")

        conf = calculate_confidence(entity)

        entity2 = Entity(id="test2", text="Test2", category="UNCORROBORATED")
        entity2.originated_from = ResponseRef("gpt-4", "r1", "q1", "q", "r")

        conf2 = calculate_confidence(entity2)

        assert conf > conf2

    def test_echo_chamber_penalty(self):
        """Echo chambers get penalized."""
        # Multi-source entity
        e1 = Entity(id="e1", text="Multi", category="UNPROCESSED")
        e1.originated_from = ResponseRef("model-a", "r1", "q1", "q", "r")
        e1.confirmed_by = [ResponseRef("model-b", "r2", "q2", "q", "r")]

        # Echo chamber entity
        e2 = Entity(id="e2", text="Echo", category="UNPROCESSED")
        e2.originated_from = ResponseRef("model-a", "r1", "q1", "q", "r")
        e2.confirmed_by = [
            ResponseRef("model-a", "r2", "q2", "q", "r"),
            ResponseRef("model-a", "r3", "q3", "q", "r")
        ]

        conf1 = calculate_confidence(e1)
        conf2 = calculate_confidence(e2)

        assert conf1 > conf2  # Echo chamber should be penalized

    def test_drill_affects_confidence(self):
        """High drill scores boost confidence."""
        entity = Entity(id="test", text="Test", category="UNCORROBORATED")
        entity.originated_from = ResponseRef("gpt-4", "r1", "q1", "q", "r")

        conf_before = calculate_confidence(entity)

        entity.drill_scores = DrillScores(
            consistency=0.9,
            contradiction=0.8,
            detail=0.7,
            provenance=0.8,
            peripheral=0.7,
            control=0.9
        )

        conf_after = calculate_confidence(entity)

        assert conf_after > conf_before

    def test_confidence_labels(self):
        """Test label assignment."""
        assert confidence_label(0.8) == "HIGH"
        assert confidence_label(0.6) == "MEDIUM"
        assert confidence_label(0.4) == "LOW"
        assert confidence_label(0.1) == "VERY LOW"


class TestDrillHelpers:
    """Test drill protocol helper functions."""

    def test_extract_key_facts(self):
        """Extract entities and dates from text."""
        text = "John Smith worked at Google in 2019 and earned $100,000"
        facts = extract_key_facts(text)

        assert "John Smith" in facts or "John" in facts
        assert "Google" in facts
        assert "2019" in facts

    def test_consistency_calculation(self):
        """Test consistency scoring."""
        # Consistent responses
        consistent = [
            "Project Aurora was created by John in 2019",
            "John created Project Aurora in 2019",
            "In 2019, John made Project Aurora"
        ]
        score = calculate_consistency(consistent)
        assert score > 0.5

        # Inconsistent responses
        inconsistent = [
            "Project Aurora was created by John in 2019",
            "Project Beta was made by Sarah in 2020",
            "Nothing happened ever"
        ]
        score2 = calculate_consistency(inconsistent)
        assert score2 < score

    def test_contradiction_resistance(self):
        """Test contradiction resistance detection."""
        # Resisting response
        resist = check_contradiction_resistance(
            "Project Aurora",
            "I thought it was Project Beta?",
            "No, it was definitely Project Aurora, not Beta."
        )

        # Folding response
        fold = check_contradiction_resistance(
            "Project Aurora",
            "I thought it was Project Beta?",
            "Yes, you're right, it was Project Beta."
        )

        assert resist > fold

    def test_detail_coherence(self):
        """Test detail coherence scoring."""
        # Natural responses (some uncertainty)
        natural = [
            "I know it was created in 2019",
            "I'm not sure about the exact date, but it was around Q3",
            "I don't have information about the contributors"
        ]
        score = check_detail_coherence(natural)

        # Overconfident responses
        overconfident = [
            "Definitely created on January 15, 2019 at 3:42 PM",
            "Absolutely certain there were exactly 5 contributors",
            "Without doubt the budget was $1,234,567.89"
        ]
        score2 = check_detail_coherence(overconfident)

        assert score > score2  # Natural should score higher

    def test_provenance_check(self):
        """Test provenance scoring."""
        # Clear source
        clear = check_provenance(
            "According to the 2019 court filing, this was documented."
        )

        # Vague source
        vague = check_provenance(
            "I just know this is true. Everyone knows it."
        )

        assert clear > vague


class TestGraph:
    """Test entity graph functionality."""

    def setup_method(self):
        """Create temp store with test data."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.store = EntityStore(self.temp_dir)

        # Add entities with relationships
        self.store.add_entity("Entity A", "model-1", "r1", "q1", "q", "Entity A and Entity B")
        self.store.add_entity("Entity B", "model-1", "r1", "q1", "q", "Entity A and Entity B")
        self.store.add_entity("Entity C", "model-2", "r2", "q2", "q", "Entity C alone")

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_graph_creation(self):
        """Graph is created from entity store."""
        graph = EntityGraph(self.store)

        assert len(graph.nodes) == 3
        assert len(graph.edges) > 0  # Should have co-occurrence edge

    def test_cooccurrence_edges(self):
        """Entities in same response are connected."""
        graph = EntityGraph(self.store)

        # A and B should be connected (same response)
        a_id = make_entity_id("Entity A")
        connections = graph.get_connections(a_id)

        target_ids = [c[0] for c in connections]
        b_id = make_entity_id("Entity B")

        assert b_id in target_ids

    def test_clusters(self):
        """Find connected clusters."""
        graph = EntityGraph(self.store)
        clusters = graph.get_clusters(min_size=2)

        # A and B should be in a cluster together
        assert len(clusters) >= 1


class TestShouldDrill:
    """Test drill prioritization."""

    def test_uncorroborated_should_drill(self):
        """Uncorroborated entities should be drilled."""
        entity = Entity(id="test", text="Test", category="UNCORROBORATED")
        entity.originated_from = ResponseRef("gpt-4", "r1", "q1", "q", "r")

        assert should_drill(entity) is True

    def test_already_drilled_skip(self):
        """Already drilled entities should be skipped."""
        entity = Entity(id="test", text="Test", category="UNCORROBORATED")
        entity.originated_from = ResponseRef("gpt-4", "r1", "q1", "q", "r")
        entity.drill_scores = DrillScores(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

        assert should_drill(entity) is False

    def test_echo_chamber_should_drill(self):
        """Echo chambers should be drilled even if 'corroborated' by frequency."""
        entity = Entity(id="test", text="Test", category="UNPROCESSED")
        entity.originated_from = ResponseRef("model-a", "r1", "q1", "q", "r")
        entity.confirmed_by = [
            ResponseRef("model-a", "r2", "q2", "q", "r"),
            ResponseRef("model-a", "r3", "q3", "q", "r")
        ]

        assert entity.is_echo_chamber
        assert should_drill(entity) is True
