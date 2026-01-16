"""
Unit test fixtures.

All unit tests should be:
- Fast (< 100ms)
- Isolated (no external dependencies)
- Deterministic (same result every time)
"""

import pytest
from datetime import datetime


@pytest.fixture
def fixed_time():
    """Fixed datetime for deterministic tests."""
    return datetime(2024, 1, 15, 12, 0, 0)


@pytest.fixture
def project_data():
    """Raw project data dict."""
    return {
        "name": "test-project",
        "topic": "Test investigation",
        "selected_models": ["model-a", "model-b"],
        "hidden_entities": ["Noise"],
        "promoted_entities": ["Signal"],
    }


@pytest.fixture
def response_data():
    """Raw probe response data."""
    return {
        "question_index": 0,
        "run_index": 0,
        "model": "test-model",
        "question": "Test question?",
        "response": "Test response mentioning John Doe and Acme Corp.",
        "entities": ["John Doe", "Acme Corp"],
        "discovered_entities": ["Acme Corp"],
        "introduced_entities": ["John Doe"],
        "is_refusal": False,
    }


@pytest.fixture
def skeptic_data():
    """Raw skeptic feedback data."""
    return {
        "weakest_link": "The connection lacks evidence",
        "alternative_explanation": "Could be coincidence",
        "circular_evidence": ["Entity A", "Entity B"],
        "counter_questions": [
            "What disproves this?",
            "Who benefits from this narrative?",
        ],
        "missing_research": "Check court records",
        "confidence": "LOW",
    }
