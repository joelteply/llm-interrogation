"""Pytest configuration for test suite."""
import sys
sys.path.insert(0, '.')

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests that call external APIs (deselect with '-m not integration')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m not slow')"
    )
