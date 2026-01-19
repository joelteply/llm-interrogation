"""
Single source of truth for the currently active project.

Everything reads from here - probe, workers, all of it.
"""

import threading

_lock = threading.Lock()
_active_project: str | None = None


def get_active_project() -> str | None:
    """Get the currently active project."""
    with _lock:
        return _active_project


def set_active_project(project_name: str | None) -> None:
    """Set the currently active project."""
    global _active_project
    with _lock:
        _active_project = project_name
        if project_name:
            print(f"[ACTIVE PROJECT] Now: {project_name}")
        else:
            print("[ACTIVE PROJECT] Cleared")
