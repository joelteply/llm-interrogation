"""
Base worker class - handles threading, lifecycle, and stats.

Concrete workers just implement _do_work().
"""

import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Callable
from datetime import datetime

from models import WorkerStats


class BaseWorker(ABC):
    """
    Base class for all background workers.

    Handles:
    - Thread lifecycle (start/stop)
    - Stats tracking
    - Callbacks for notifications
    - Active project tracking

    Subclasses implement _do_work() with their actual logic.
    """

    def __init__(self, name: str, interval: float = 60.0):
        self.name = name
        self.interval = interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callbacks: list[Callable] = []
        self._active_project: Optional[str] = None
        self.stats = WorkerStats()

    def set_project(self, project_name: Optional[str]) -> None:
        """Set the active project. None to pause/clear."""
        self._active_project = project_name
        self.stats.active_project = project_name
        if project_name:
            print(f"[{self.name}] Focused on: {project_name}")

    def start(self) -> None:
        """Start the worker thread."""
        if self._thread and self._thread.is_alive():
            print(f"[{self.name}] Already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[{self.name}] Started (interval={self.interval}s)")

    def stop(self) -> None:
        """Stop the worker thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                print(f"[{self.name}] WARNING: Thread didn't stop cleanly")
        self._thread = None
        print(f"[{self.name}] Stopped")

    def is_running(self) -> bool:
        """Check if worker is active."""
        return self._thread is not None and self._thread.is_alive()

    def add_callback(self, callback: Callable) -> None:
        """Add callback for notifications."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def notify(self, event_type: str, data: dict) -> None:
        """Notify all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(event_type, data)
            except Exception as e:
                print(f"[{self.name}] Callback error: {e}")

    def _run_loop(self) -> None:
        """Main worker loop."""
        print(f"[{self.name}] Loop started")

        while not self._stop_event.is_set():
            project = self._get_project()

            if project:
                self.stats.record_run(project)
                try:
                    items = self._do_work(project)
                    self.stats.record_success(items)
                except Exception as e:
                    print(f"[{self.name}] Error: {e}")
                    self.stats.record_error(str(e))

            self._stop_event.wait(self.interval)

        print(f"[{self.name}] Loop ended")

    def _get_project(self) -> Optional[str]:
        """Get project to work on. Override for custom logic."""
        if self._active_project:
            return self._active_project

        # Default: pick most recently updated
        from routes import project_storage as storage
        projects = storage.list_projects()
        if projects:
            first = projects[0]
            return first.get("name") if isinstance(first, dict) else first
        return None

    @abstractmethod
    def _do_work(self, project_name: str) -> int:
        """
        Do the actual work.

        Args:
            project_name: Project to work on

        Returns:
            Number of items processed (for stats)
        """
        pass

    def get_stats(self) -> dict:
        """Get stats as dict for API."""
        return {
            "name": self.name,
            "running": self.is_running(),
            **self.stats.to_dict(),
        }
