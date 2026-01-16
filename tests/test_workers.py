"""
Tests for background workers.

Run: ./venv/bin/python -m pytest tests/test_workers.py -v
"""

import pytest
import time
from unittest.mock import MagicMock, patch


class TestResearchWorker:
    """Test research worker."""

    def test_worker_creation(self):
        from workers.research import ResearchWorker
        worker = ResearchWorker(interval=60.0, max_queries_per_run=5)
        assert worker.interval == 60.0
        assert worker.max_queries_per_run == 5
        assert not worker.is_running()

    def test_set_project(self):
        from workers.research import ResearchWorker
        worker = ResearchWorker()
        worker.set_project("test-project")
        assert worker._active_project == "test-project"

    def test_start_stop(self):
        from workers.research import ResearchWorker
        worker = ResearchWorker(interval=1.0)
        worker.start()
        assert worker.is_running()
        worker.stop()
        time.sleep(0.1)
        assert not worker.is_running()

    def test_global_worker(self):
        from workers.research import get_worker, start_worker, stop_worker
        worker = get_worker()
        assert worker is not None
        # Should return same instance
        assert get_worker() is worker


class TestSkepticWorker:
    """Test skeptic (devil's advocate) worker."""

    def test_worker_creation(self):
        from workers.skeptic import SkepticWorker
        worker = SkepticWorker(interval=90.0)
        assert worker.interval == 90.0
        assert not worker.is_running()

    def test_set_project(self):
        from workers.skeptic import SkepticWorker
        worker = SkepticWorker()
        worker.set_project("test-project")
        assert worker._active_project == "test-project"

    def test_start_stop(self):
        from workers.skeptic import SkepticWorker
        worker = SkepticWorker(interval=1.0)
        worker.start()
        assert worker.is_running()
        worker.stop()
        time.sleep(0.1)
        assert not worker.is_running()

    def test_global_worker(self):
        from workers.skeptic import get_worker, start_worker, stop_worker
        worker = get_worker()
        assert worker is not None
        assert get_worker() is worker

    def test_parse_skeptic_response(self):
        """Test parsing of LLM skeptic response."""
        from workers.skeptic import SkepticWorker
        worker = SkepticWorker()

        response = """WEAKEST_LINK: The connection between X and Y has only one source.

ALTERNATIVE_EXPLANATION: This could simply be a coincidence or misattribution.

CIRCULAR_EVIDENCE:
- Entity A (only mentioned because we asked about A)
- Entity B (appears in question text)

COUNTER_QUESTIONS:
- What evidence contradicts the X-Y connection?
- Are there any sources that dispute this claim?
- Who benefits from this narrative being true?

MISSING_RESEARCH: No court records have been checked.

CONFIDENCE_ASSESSMENT: LOW"""

        result = worker._parse_skeptic_response(response)

        assert "connection between X and Y" in result["weakest_link"]
        assert "coincidence" in result["alternative_explanation"]
        assert len(result["circular_evidence"]) == 2
        assert len(result["counter_questions"]) == 3
        assert "court records" in result["missing_research"]
        assert result["confidence"] == "LOW"


class TestEntityVerificationWorker:
    """Test entity verification worker."""

    def test_worker_creation(self):
        from workers.entity_verification import EntityVerificationWorker
        worker = EntityVerificationWorker(interval=45.0, max_entities_per_run=10)
        assert worker.interval == 45.0
        assert worker.max_entities_per_run == 10

    def test_start_stop(self):
        from workers.entity_verification import EntityVerificationWorker
        worker = EntityVerificationWorker(interval=1.0)
        worker.start()
        assert worker.is_running()
        worker.stop()
        time.sleep(0.1)
        assert not worker.is_running()


class TestWebAnglesWorker:
    """Test web angles worker."""

    def test_worker_creation(self):
        from workers.web_angles import WebAnglesWorker
        worker = WebAnglesWorker(interval=60.0)
        assert worker.interval == 60.0

    def test_start_stop(self):
        from workers.web_angles import WebAnglesWorker
        worker = WebAnglesWorker(interval=1.0)
        worker.start()
        assert worker.is_running()
        worker.stop()
        time.sleep(0.1)
        assert not worker.is_running()


class TestWorkersModule:
    """Test workers module functions."""

    def test_get_all_stats(self):
        from workers import get_all_stats
        stats = get_all_stats()

        assert "research" in stats
        assert "web_angles" in stats
        assert "entity_verification" in stats
        assert "skeptic" in stats

        for name, worker_stats in stats.items():
            assert "running" in worker_stats
            assert "errors" in worker_stats

    def test_start_stop_all(self):
        from workers import start_all_workers, stop_all_workers, get_all_stats

        start_all_workers()
        time.sleep(0.5)
        stats = get_all_stats()

        # All should be running
        for name, worker_stats in stats.items():
            assert worker_stats["running"], f"{name} should be running"

        stop_all_workers()
        time.sleep(6)  # Workers have 5s join timeout
        stats = get_all_stats()

        # All should be stopped
        for name, worker_stats in stats.items():
            assert not worker_stats["running"], f"{name} should be stopped"


class TestCleanerFunctions:
    """Test research cleaner functions."""

    def test_is_useful_research_rejects_blocked_pages(self):
        from routes.analyze.research.cleaner import is_useful_research

        # Test heuristic rejection (doesn't need LLM)
        blocked_content = "Just a moment... Checking your browser before accessing the site. " * 10
        is_useful, reason = is_useful_research(blocked_content, "Test Page", "test topic")
        assert not is_useful
        assert "Blocked" in reason or "just a moment" in reason.lower()

    def test_is_useful_research_rejects_short_content(self):
        from routes.analyze.research.cleaner import is_useful_research

        short_content = "Too short"
        is_useful, reason = is_useful_research(short_content, "Test", "topic")
        assert not is_useful
        assert "Too short" in reason

    def test_looks_like_garbage(self):
        from routes.analyze.research.cleaner import looks_like_garbage

        clean = "This is normal text without any garbage characters."
        assert not looks_like_garbage(clean)

        garbage = "Th\ufffdis h\ufffds g\ufffdrb\ufffdge ch\ufffdr\ufffdcters" * 10
        assert looks_like_garbage(garbage)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
