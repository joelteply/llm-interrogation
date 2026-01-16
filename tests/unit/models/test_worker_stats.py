"""Unit tests for WorkerStats model."""

import pytest
from models import WorkerStats


class TestWorkerStats:
    """Test WorkerStats model."""

    def test_create_default(self):
        stats = WorkerStats()
        assert stats.runs == 0
        assert stats.successes == 0
        assert stats.errors == 0
        assert stats.last_run is None

    def test_record_run(self):
        stats = WorkerStats()
        stats.record_run("test-project")

        assert stats.runs == 1
        assert stats.last_run is not None
        assert stats.last_project == "test-project"

    def test_record_success(self):
        stats = WorkerStats()
        stats.record_success(items=5)

        assert stats.successes == 1
        assert stats.items_processed == 5
        assert stats.last_success is not None

    def test_record_error(self):
        stats = WorkerStats()
        stats.record_error("Something failed")

        assert stats.errors == 1
        assert stats.last_error is not None
        assert stats.last_error_message == "Something failed"

    def test_success_rate_no_runs(self):
        stats = WorkerStats()
        assert stats.success_rate == 0.0

    def test_success_rate_calculated(self):
        stats = WorkerStats(runs=10, successes=8)
        assert stats.success_rate == 0.8

    def test_is_healthy_not_enough_data(self):
        stats = WorkerStats(runs=2)
        assert stats.is_healthy  # Not enough data to judge

    def test_is_healthy_good_rate(self):
        stats = WorkerStats(runs=10, successes=8, errors=2)
        assert stats.is_healthy

    def test_is_healthy_bad_rate(self):
        stats = WorkerStats(runs=10, successes=4, errors=6)
        assert not stats.is_healthy

    def test_to_dict(self):
        stats = WorkerStats(runs=5, successes=4, errors=1)
        stats.active_project = "test"

        d = stats.to_dict()

        assert d["runs"] == 5
        assert d["successes"] == 4
        assert d["errors"] == 1
        assert d["active_project"] == "test"
        assert "healthy" in d
