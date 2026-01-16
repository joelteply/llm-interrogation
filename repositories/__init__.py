"""
Repository layer - abstracts persistence.

Usage:
    from repositories import get_repository

    repo = get_repository()  # Returns configured backend
    project = repo.projects.get("my-project")
    repo.projects.save(project)

Backends are swappable via config.
"""

from .base import Repository
from .json_backend import JsonRepository

# Default backend - can be changed via config
_backend: str = "json"
_instance: Repository = None


def get_repository() -> Repository:
    """Get the configured repository instance."""
    global _instance

    if _instance is None:
        if _backend == "json":
            _instance = JsonRepository()
        # Add more backends here:
        # elif _backend == "sqlite":
        #     _instance = SqliteRepository()
        # elif _backend == "rest":
        #     _instance = RestRepository()
        else:
            raise ValueError(f"Unknown backend: {_backend}")

    return _instance


def configure_backend(backend: str, **kwargs) -> None:
    """Configure the repository backend."""
    global _backend, _instance
    _backend = backend
    _instance = None  # Force re-initialization


__all__ = ["get_repository", "configure_backend", "Repository"]
