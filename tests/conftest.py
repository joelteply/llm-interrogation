"""
Root test configuration.

Test organization:
- unit/        Fast, isolated, no I/O, mocked dependencies
- integration/ Component boundaries, real I/O to temp locations
- e2e/         Full system tests

Run specific levels:
    pytest tests/unit -v           # Fast feedback loop
    pytest tests/integration -v    # Before commit
    pytest tests/e2e -v            # CI/CD
    pytest tests -v                # Everything
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Fast isolated tests")
    config.addinivalue_line("markers", "integration: Component boundary tests")
    config.addinivalue_line("markers", "e2e: End-to-end system tests")
    config.addinivalue_line("markers", "slow: Tests that take > 1s")


@pytest.fixture
def sample_topic():
    """Standard test topic."""
    return "Test investigation topic"


@pytest.fixture
def sample_entities():
    """Standard test entities."""
    return ["John Doe", "Acme Corp", "New York", "2024-01-15"]


# === Performance tracking ===

@pytest.fixture
def benchmark(request):
    """
    Simple benchmark fixture.

    Usage:
        def test_something(benchmark):
            result = benchmark(my_function, arg1, arg2)
            assert result == expected
    """
    import time

    def run(fn, *args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start

        # Log timing (visible with pytest -v)
        test_name = request.node.name
        print(f"\n  [{test_name}] {elapsed*1000:.2f}ms")

        return result

    return run


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add timing summary at end of test run."""
    stats = terminalreporter.stats

    # Collect slowest tests
    if 'passed' in stats:
        durations = []
        for report in stats['passed']:
            if hasattr(report, 'duration'):
                durations.append((report.duration, report.nodeid))

        if durations:
            durations.sort(reverse=True)
            terminalreporter.write_sep("=", "slowest 5 tests")
            for duration, nodeid in durations[:5]:
                terminalreporter.write_line(f"  {duration:.2f}s  {nodeid}")
