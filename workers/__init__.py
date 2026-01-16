"""
Background workers for non-blocking operations.

Workers run continuously and poll for work, updating project metadata
which the main probe reads from cache.
"""

from .research import ResearchWorker

__all__ = ['ResearchWorker']
