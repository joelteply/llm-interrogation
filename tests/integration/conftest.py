"""
Integration test fixtures.

Integration tests:
- Test component boundaries
- Use real I/O but to temp locations
- Should be deterministic
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Temporary directory for test data."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def projects_dir(temp_dir):
    """Temporary projects directory."""
    p = temp_dir / "projects"
    p.mkdir()
    return p
