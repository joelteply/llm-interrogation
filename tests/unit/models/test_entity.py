"""Unit tests for Entity model."""

import pytest
from models import Entity, EntityType, EntityCategory
from models.entity import Provenance, Relationship


class TestEntity:
    """Test Entity model."""

    def test_create_from_text(self):
        entity = Entity.from_text("John Doe")
        assert entity.text == "John Doe"
        assert entity.normalized == "john doe"
        assert entity.id == "john_doe"

    def test_create_with_custom_id(self):
        entity = Entity.from_text("John Doe", id="person_123")
        assert entity.id == "person_123"
        assert entity.text == "John Doe"

    def test_create_with_type(self):
        entity = Entity.from_text("Acme Corp", entity_type=EntityType.ORGANIZATION)
        assert entity.entity_type == EntityType.ORGANIZATION

    def test_default_category_is_unprocessed(self):
        entity = Entity.from_text("Test")
        assert entity.category == EntityCategory.UNPROCESSED

    def test_confidence_bounds(self):
        entity = Entity.from_text("Test", confidence=0.5)
        assert entity.confidence == 0.5

        with pytest.raises(ValueError):
            Entity.from_text("Test", confidence=1.5)

        with pytest.raises(ValueError):
            Entity.from_text("Test", confidence=-0.1)

    def test_whitespace_stripped(self):
        entity = Entity.from_text("  John Doe  ")
        assert entity.text == "John Doe"
        assert entity.normalized == "john doe"

    def test_timestamps_set_on_create(self):
        entity = Entity.from_text("Test")
        assert entity.created_at is not None
        assert entity.updated_at is not None


class TestProvenance:
    """Test Provenance model."""

    def test_create_provenance(self):
        prov = Provenance(
            model="gpt-4",
            question="Who is involved?",
            response_snippet="John Doe was mentioned..."
        )
        assert prov.model == "gpt-4"
        assert prov.timestamp is not None


class TestRelationship:
    """Test Relationship model."""

    def test_create_relationship(self):
        rel = Relationship(
            target_id="acme_corp",
            relation_type="works_at",
            confidence=0.8
        )
        assert rel.target_id == "acme_corp"
        assert rel.relation_type == "works_at"
        assert rel.confidence == 0.8

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            Relationship(target_id="x", relation_type="y", confidence=1.5)
