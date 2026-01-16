"""
Worker models - shared across all background workers.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class WorkerStats(BaseModel):
    """
    Statistics for a background worker.

    Single definition - no more duplicates in each worker file.
    """
    # Counters (worker-specific, use generically)
    runs: int = 0
    successes: int = 0
    errors: int = 0

    # Specific counters (optional, workers set what they need)
    items_processed: int = 0
    items_cached: int = 0
    items_skipped: int = 0

    # Timing
    last_run: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_error: Optional[datetime] = None
    last_error_message: Optional[str] = None

    # Context
    last_project: Optional[str] = None
    active_project: Optional[str] = None

    def record_run(self, project: str = None) -> None:
        """Record a run attempt."""
        self.runs += 1
        self.last_run = datetime.now()
        if project:
            self.last_project = project

    def record_success(self, items: int = 0) -> None:
        """Record successful run."""
        self.successes += 1
        self.last_success = datetime.now()
        self.items_processed += items

    def record_error(self, message: str = None) -> None:
        """Record an error."""
        self.errors += 1
        self.last_error = datetime.now()
        self.last_error_message = message

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.runs == 0:
            return 0.0
        return self.successes / self.runs

    @property
    def is_healthy(self) -> bool:
        """Check if worker is healthy (recent success, low error rate)."""
        if self.runs < 3:
            return True  # Not enough data
        return self.success_rate > 0.5

    def to_dict(self) -> dict:
        """Export for API responses."""
        return {
            "runs": self.runs,
            "successes": self.successes,
            "errors": self.errors,
            "items_processed": self.items_processed,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_error": self.last_error_message,
            "active_project": self.active_project,
            "healthy": self.is_healthy,
        }
