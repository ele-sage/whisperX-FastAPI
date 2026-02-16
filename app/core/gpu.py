"""GPU access serialization for preventing OOM under concurrent requests.

Uses per-stage semaphores so different pipeline stages (transcription,
alignment, diarization) can run in parallel on different models, while
preventing two tasks from using the same heavy model simultaneously.
"""

import threading
from contextlib import contextmanager
from typing import Generator

from app.core import get_settings
from app.core.logging import logger

_transcription_lock = threading.Semaphore(get_settings().whisper.TRANSCRIPTION_SEMAPHORE_LIMIT)
_alignment_lock = threading.Semaphore(get_settings().whisper.ALIGNMENT_SEMAPHORE_LIMIT)
_diarization_lock = threading.Semaphore(get_settings().whisper.DIARIZATION_SEMAPHORE_LIMIT)

@contextmanager
def gpu_lock(stage: str) -> Generator[None, None, None]:
    """
    Context manager that serializes GPU access per pipeline stage.

    Args:
        stage: One of 'transcription', 'alignment', 'diarization'.

    Usage::

        with gpu_lock("transcription"):
            model.transcribe(audio)
    """
    sem = _get_semaphore(stage)
    logger.debug("Waiting to acquire GPU lock for %s...", stage)
    sem.acquire()
    logger.debug("GPU lock acquired for %s.", stage)
    try:
        yield
    finally:
        sem.release()
        logger.debug("GPU lock released for %s.", stage)


def _get_semaphore(stage: str) -> threading.Semaphore:
    """Return the semaphore for the given pipeline stage."""
    if stage == "transcription":
        return _transcription_lock
    elif stage == "alignment":
        return _alignment_lock
    elif stage == "diarization":
        return _diarization_lock
    else:
        raise ValueError(f"Unknown GPU stage: {stage!r}")


