"""Unit tests for Project model."""

import pytest
from models import Project
from models.corpus import Question


class TestProject:
    """Test Project model."""

    def test_create_minimal(self):
        project = Project(name="test")
        assert project.name == "test"
        assert project.topic == ""
        assert project.selected_models == []

    def test_create_full(self, project_data):
        project = Project(**project_data)
        assert project.name == "test-project"
        assert project.topic == "Test investigation"
        assert len(project.selected_models) == 2

    def test_hide_entity(self):
        project = Project(name="test")
        project.hide_entity("Noise")
        assert "Noise" in project.hidden_entities

    def test_hide_entity_removes_from_promoted(self):
        project = Project(name="test", promoted_entities=["Noise"])
        project.hide_entity("Noise")
        assert "Noise" in project.hidden_entities
        assert "Noise" not in project.promoted_entities

    def test_promote_entity(self):
        project = Project(name="test")
        project.promote_entity("Signal")
        assert "Signal" in project.promoted_entities

    def test_promote_entity_removes_from_hidden(self):
        project = Project(name="test", hidden_entities=["Signal"])
        project.promote_entity("Signal")
        assert "Signal" in project.promoted_entities
        assert "Signal" not in project.hidden_entities

    def test_add_angle(self):
        project = Project(name="test")
        project.add_angle("Financial connections")
        assert "Financial connections" in project.angles

    def test_add_angle_no_duplicates(self):
        project = Project(name="test")
        project.add_angle("Financial connections")
        project.add_angle("Financial connections")
        assert project.angles.count("Financial connections") == 1

    def test_update_narrative(self):
        project = Project(name="test")
        project.update_narrative("New theory")
        assert project.narrative == "New theory"
        assert project.narrative_updated is not None

    def test_touch_updates_timestamp(self):
        project = Project(name="test")
        original = project.updated_at
        project.touch()
        assert project.updated_at >= original

    def test_allows_extra_fields_for_migration(self):
        """Legacy data may have extra fields - should not fail."""
        project = Project(
            name="test",
            topic="Test",
            legacy_field="should be ignored"
        )
        assert project.name == "test"


class TestQuestion:
    """Test Question model."""

    def test_create_minimal(self):
        q = Question(text="What happened?")
        assert q.text == "What happened?"
        assert q.technique == "direct"

    def test_create_from_question_alias(self):
        """Legacy data uses 'question' not 'text'."""
        q = Question(question="What happened?")
        assert q.text == "What happened?"

    def test_question_property(self):
        q = Question(text="What happened?")
        assert q.question == "What happened?"

    def test_with_technique(self):
        q = Question(text="Timeline?", technique="timeline")
        assert q.technique == "timeline"

    def test_with_color(self):
        q = Question(text="Test", color="#ff0000")
        assert q.color == "#ff0000"
