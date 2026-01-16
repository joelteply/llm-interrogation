"""Unit tests for BaseWorker."""

import pytest
import time
from unittest.mock import MagicMock, patch

from workers.base import BaseWorker


class ConcreteWorker(BaseWorker):
    """Concrete implementation for testing."""

    def __init__(self, **kwargs):
        super().__init__(name="TEST-WORKER", **kwargs)
        self.work_calls = []
        self.work_return = 1
        self.work_error = None

    def _do_work(self, project_name: str) -> int:
        self.work_calls.append(project_name)
        if self.work_error:
            raise self.work_error
        return self.work_return


class TestBaseWorker:
    """Test BaseWorker functionality."""

    def test_create_worker(self):
        worker = ConcreteWorker(interval=30.0)
        assert worker.name == "TEST-WORKER"
        assert worker.interval == 30.0
        assert not worker.is_running()

    def test_set_project(self):
        worker = ConcreteWorker()
        worker.set_project("test-project")
        assert worker._active_project == "test-project"
        assert worker.stats.active_project == "test-project"

    def test_set_project_none_clears(self):
        worker = ConcreteWorker()
        worker.set_project("test")
        worker.set_project(None)
        assert worker._active_project is None

    def test_start_stop(self):
        worker = ConcreteWorker(interval=0.1)
        worker.set_project("test")

        worker.start()
        assert worker.is_running()

        time.sleep(0.05)
        worker.stop()
        assert not worker.is_running()

    def test_stats_recorded_on_work(self):
        worker = ConcreteWorker(interval=0.1)
        worker.set_project("test")
        worker.work_return = 5

        worker.start()
        time.sleep(0.15)  # Allow one cycle
        worker.stop()

        assert worker.stats.runs >= 1
        assert worker.stats.successes >= 1
        assert worker.stats.items_processed >= 5

    def test_error_recorded_on_failure(self):
        worker = ConcreteWorker(interval=0.1)
        worker.set_project("test")
        worker.work_error = ValueError("Test error")

        worker.start()
        time.sleep(0.15)
        worker.stop()

        assert worker.stats.errors >= 1
        assert "Test error" in (worker.stats.last_error_message or "")

    def test_callbacks_notified(self):
        worker = ConcreteWorker(interval=0.1)
        callback = MagicMock()
        worker.add_callback(callback)

        worker.notify("test_event", {"key": "value"})

        callback.assert_called_once_with("test_event", {"key": "value"})

    def test_remove_callback(self):
        worker = ConcreteWorker()
        callback = MagicMock()
        worker.add_callback(callback)
        worker.remove_callback(callback)

        worker.notify("test_event", {})

        callback.assert_not_called()

    def test_callback_error_doesnt_crash(self):
        worker = ConcreteWorker()
        bad_callback = MagicMock(side_effect=Exception("Callback failed"))
        worker.add_callback(bad_callback)

        # Should not raise
        worker.notify("test_event", {})

    def test_get_stats(self):
        worker = ConcreteWorker()
        worker.stats.runs = 5
        worker.stats.successes = 4

        stats = worker.get_stats()

        assert stats["name"] == "TEST-WORKER"
        assert stats["running"] is False
        assert stats["runs"] == 5
        assert stats["successes"] == 4

    def test_default_project_selection(self):
        """Without active project, picks most recent."""
        worker = ConcreteWorker(interval=0.1)

        with patch('routes.project_storage.list_projects') as mock_list:
            mock_list.return_value = [
                {"name": "recent-project"},
                {"name": "old-project"}
            ]

            worker.start()
            time.sleep(0.15)
            worker.stop()

            # Should have worked on recent-project
            assert "recent-project" in worker.work_calls
