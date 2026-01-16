"""Unit tests for analysis models."""

import pytest
from models import SkepticFeedback, EntityVerification
from models.analysis import ConfidenceLevel, VerifiedEntity


class TestSkepticFeedback:
    """Test SkepticFeedback model."""

    def test_create_empty(self):
        feedback = SkepticFeedback()
        assert feedback.confidence == "LOW"
        assert feedback.counter_questions == []

    def test_create_full(self, skeptic_data):
        feedback = SkepticFeedback(**skeptic_data)
        assert feedback.weakest_link == "The connection lacks evidence"
        assert len(feedback.counter_questions) == 2

    def test_has_content_false_when_empty(self):
        feedback = SkepticFeedback()
        assert not feedback.has_content

    def test_has_content_true_with_weakest_link(self):
        feedback = SkepticFeedback(weakest_link="Something weak")
        assert feedback.has_content

    def test_has_content_true_with_questions(self):
        feedback = SkepticFeedback(counter_questions=["Question?"])
        assert feedback.has_content

    def test_confidence_level_extracts_from_verbose(self):
        """Legacy data has verbose confidence strings."""
        feedback = SkepticFeedback(confidence="LOW. The theory relies on assumptions.")
        assert feedback.confidence_level == ConfidenceLevel.LOW

    def test_confidence_level_medium(self):
        feedback = SkepticFeedback(confidence="MEDIUM - needs more evidence")
        assert feedback.confidence_level == ConfidenceLevel.MEDIUM

    def test_confidence_level_high(self):
        feedback = SkepticFeedback(confidence="HIGH")
        assert feedback.confidence_level == ConfidenceLevel.HIGH

    def test_handles_updated_alias(self):
        """Legacy data uses 'updated' not 'updated_at'."""
        feedback = SkepticFeedback(updated=1234567890.0)
        assert feedback.updated_at is not None


class TestEntityVerification:
    """Test EntityVerification model."""

    def test_create_empty(self):
        v = EntityVerification()
        assert v.verified == []
        assert v.unverified == []
        assert v.unknown == []

    def test_mark_verified(self):
        v = EntityVerification(unknown=["John Doe"])
        v.mark_verified("John Doe", url="http://example.com", snippet="Found")

        assert "John Doe" not in v.unknown
        assert len(v.verified) == 1
        assert v.verified[0].entity == "John Doe"
        assert v.verified[0].url == "http://example.com"

    def test_mark_verified_moves_from_unverified(self):
        v = EntityVerification()
        v.unverified.append(VerifiedEntity(entity="John Doe"))
        v.mark_verified("John Doe", url="http://example.com")

        assert len(v.unverified) == 0
        assert len(v.verified) == 1

    def test_mark_unverified(self):
        v = EntityVerification(unknown=["Mystery Person"])
        v.mark_unverified("Mystery Person")

        assert "Mystery Person" not in v.unknown
        assert len(v.unverified) == 1
        assert v.unverified[0].entity == "Mystery Person"

    def test_public_entities(self):
        v = EntityVerification()
        v.verified.append(VerifiedEntity(entity="John Doe"))
        v.verified.append(VerifiedEntity(entity="Jane Doe"))

        assert v.public_entities == ["John Doe", "Jane Doe"]

    def test_private_entities(self):
        v = EntityVerification()
        v.unverified.append(VerifiedEntity(entity="Secret Person"))

        assert v.private_entities == ["Secret Person"]

    def test_no_duplicate_verified(self):
        v = EntityVerification()
        v.mark_verified("John Doe")
        v.mark_verified("John Doe")  # Should not duplicate

        assert len(v.verified) == 1
