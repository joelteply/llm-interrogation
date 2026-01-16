"""
Thread-safe write queue for project storage.

All project metadata writes go through this queue to prevent race conditions
from multiple workers writing to the same file simultaneously.
"""

import json
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional


class WriteQueue:
    """Serializes all file writes through a single thread."""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    def start(self):
        """Start the write queue worker thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0):
        """Stop the write queue, processing remaining items."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        # Signal thread to stop
        self._queue.put(None)

        if self._thread:
            self._thread.join(timeout=timeout)

    def _worker(self):
        """Worker thread that processes write operations."""
        while True:
            try:
                item = self._queue.get(timeout=1.0)
                if item is None:
                    # Shutdown signal
                    break

                file_path, data, callback = item
                try:
                    self._do_write(file_path, data)
                    if callback:
                        callback(None)
                except Exception as e:
                    print(f"[WriteQueue] Error writing {file_path}: {e}")
                    if callback:
                        callback(e)
                finally:
                    self._queue.task_done()

            except queue.Empty:
                if not self._running:
                    break
                continue

    def _do_write(self, file_path: Path, data: dict):
        """Actually write the file (called from worker thread only)."""
        # Write to temp file first, then atomic rename
        temp_path = file_path.with_suffix('.json.tmp')

        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()

        # Atomic rename (on POSIX systems)
        temp_path.replace(file_path)

    def enqueue(self, file_path: Path, data: dict, callback: Callable = None):
        """Queue a write operation."""
        if not self._running:
            self.start()

        self._queue.put((file_path, data, callback))

    def enqueue_sync(self, file_path: Path, data: dict, timeout: float = 10.0) -> bool:
        """Queue a write and wait for completion."""
        if not self._running:
            self.start()

        done_event = threading.Event()
        error_holder = [None]

        def on_complete(err):
            error_holder[0] = err
            done_event.set()

        self._queue.put((file_path, data, on_complete))

        if done_event.wait(timeout=timeout):
            if error_holder[0]:
                raise error_holder[0]
            return True
        else:
            raise TimeoutError(f"Write to {file_path} timed out after {timeout}s")

    def flush(self, timeout: float = 30.0):
        """Wait for all queued writes to complete."""
        self._queue.join()


# Global instance
_write_queue: Optional[WriteQueue] = None
_init_lock = threading.Lock()


def get_write_queue() -> WriteQueue:
    """Get or create the global write queue."""
    global _write_queue
    with _init_lock:
        if _write_queue is None:
            _write_queue = WriteQueue()
            _write_queue.start()
        return _write_queue


def queue_write(file_path: Path, data: dict):
    """Queue a write operation (fire and forget)."""
    get_write_queue().enqueue(file_path, data)


def queue_write_sync(file_path: Path, data: dict, timeout: float = 10.0):
    """Queue a write operation and wait for completion."""
    get_write_queue().enqueue_sync(file_path, data, timeout)


def shutdown_write_queue():
    """Shutdown the write queue gracefully."""
    global _write_queue
    with _init_lock:
        if _write_queue:
            _write_queue.stop()
            _write_queue = None
