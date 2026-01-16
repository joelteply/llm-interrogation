"""
Background workers for probe operations.

Clean concurrent architecture:
- WorkerPool manages all background tasks
- Queue-based communication
- Proper shutdown handling
"""

import threading
import queue
from typing import Callable, Optional
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    RESEARCH = "research"
    SYNTHESIS = "synthesis"
    VERIFY = "verify"


@dataclass
class Task:
    type: TaskType
    func: Callable
    args: tuple = ()
    kwargs: dict = None
    callback: Optional[Callable] = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


class WorkerPool:
    """
    Manages background workers for probe operations.

    Usage:
        pool = WorkerPool()
        pool.start()

        # Submit work
        pool.submit(TaskType.RESEARCH, fetch_docs, args=(query,))
        pool.submit(TaskType.SYNTHESIS, synthesize, args=(project,), callback=on_done)

        # Shutdown
        pool.stop()
    """

    def __init__(self, num_workers: int = 3):
        self.num_workers = num_workers
        self._queue = queue.Queue()
        self._workers = []
        self._stop_event = threading.Event()
        self._running = False

        # Track in-progress tasks by type (only one synthesis at a time, etc.)
        self._in_progress = {t: threading.Event() for t in TaskType}

    def start(self):
        """Start worker threads."""
        if self._running:
            return

        self._stop_event.clear()
        self._running = True

        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, name=f"Worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)

        print(f"[WORKERS] Started {self.num_workers} workers")

    def stop(self, timeout: float = 5.0):
        """Stop all workers gracefully."""
        if not self._running:
            return

        self._stop_event.set()

        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        # Wait for workers
        for t in self._workers:
            t.join(timeout=timeout)

        self._workers.clear()
        self._running = False
        print("[WORKERS] Stopped")

    def submit(self, task_type: TaskType, func: Callable, args: tuple = (), kwargs: dict = None, callback: Callable = None) -> bool:
        """
        Submit a task. Returns False if task type already in progress.
        """
        if not self._running:
            print(f"[WORKERS] Not running, can't submit {task_type}")
            return False

        # Check if this task type is already running
        if self._in_progress[task_type].is_set():
            print(f"[WORKERS] {task_type.value} already in progress, skipping")
            return False

        task = Task(type=task_type, func=func, args=args, kwargs=kwargs or {}, callback=callback)
        self._queue.put(task)
        return True

    def _worker_loop(self):
        """Worker thread main loop."""
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Mark task type as in-progress
            self._in_progress[task.type].set()

            try:
                print(f"[WORKERS] Running {task.type.value}...")
                result = task.func(*task.args, **task.kwargs)

                if task.callback:
                    try:
                        task.callback(result)
                    except Exception as e:
                        print(f"[WORKERS] Callback error: {e}")

            except Exception as e:
                print(f"[WORKERS] Task {task.type.value} failed: {e}")

            finally:
                self._in_progress[task.type].clear()
                self._queue.task_done()


# Global worker pool instance
_pool: Optional[WorkerPool] = None


def get_worker_pool() -> WorkerPool:
    """Get or create the global worker pool."""
    global _pool
    if _pool is None:
        _pool = WorkerPool(num_workers=3)
    return _pool


def submit_research(func: Callable, *args, callback: Callable = None, **kwargs) -> bool:
    """Submit a research task."""
    pool = get_worker_pool()
    if not pool._running:
        pool.start()
    return pool.submit(TaskType.RESEARCH, func, args=args, kwargs=kwargs, callback=callback)


def submit_synthesis(func: Callable, *args, callback: Callable = None, **kwargs) -> bool:
    """Submit a synthesis task."""
    pool = get_worker_pool()
    if not pool._running:
        pool.start()
    return pool.submit(TaskType.SYNTHESIS, func, args=args, kwargs=kwargs, callback=callback)


def submit_verify(func: Callable, *args, callback: Callable = None, **kwargs) -> bool:
    """Submit a verification task."""
    pool = get_worker_pool()
    if not pool._running:
        pool.start()
    return pool.submit(TaskType.VERIFY, func, args=args, kwargs=kwargs, callback=callback)


def shutdown_workers():
    """Shutdown the global worker pool."""
    global _pool
    if _pool:
        _pool.stop()
        _pool = None
