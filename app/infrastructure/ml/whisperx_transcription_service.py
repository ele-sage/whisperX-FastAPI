"""WhisperX implementation of transcription service."""

import gc
from typing import Any

import numpy as np
import torch
from whisperx import load_model

from app.core.gpu import gpu_lock
from app.core.logging import logger


class WhisperXTranscriptionService:
    """
    WhisperX-based implementation of transcription service.

    This service wraps the WhisperX library to provide transcription
    functionality following the ITranscriptionService interface contract.

    The model is lazily loaded on first call and cached for reuse.
    If key parameters change between calls the model is reloaded automatically.
    GPU access is serialized via a semaphore so concurrent requests queue safely.
    """

    def __init__(self) -> None:
        """Initialize the transcription service."""
        self.model: Any = None
        self._model_config: dict[str, Any] | None = None
        self.logger = logger

    def _should_reload(
        self,
        model: str,
        device: str,
        device_index: int,
        compute_type: str,
        language: str,
        task: str,
    ) -> bool:
        """Check whether the cached model needs to be reloaded."""
        if self.model is None or self._model_config is None:
            return True
        return self._model_config != {
            "model": model,
            "device": device,
            "device_index": device_index,
            "compute_type": compute_type,
            "language": language,
            "task": task,
        }

    def transcribe(
        self,
        audio: np.ndarray[Any, np.dtype[np.float32]],
        task: str,
        asr_options: dict[str, Any],
        vad_options: dict[str, Any],
        language: str,
        batch_size: int,
        chunk_size: int,
        model: str,
        device: str,
        device_index: int,
        compute_type: str,
        threads: int,
    ) -> dict[str, Any]:
        """
        Transcribe audio using WhisperX model.

        The model is loaded once and reused. If key parameters change
        the model is reloaded automatically. GPU access is serialized.
        """
        with gpu_lock("transcription"):
            self.logger.debug(
                "Starting transcription with Whisper model: %s on device: %s",
                model,
                device,
            )

            # Log GPU memory
            if torch.cuda.is_available():
                self.logger.debug(
                    f"GPU memory - used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
                    f"total: {torch.cuda.get_device_properties(0).total_mem / 1024**2:.2f} MB"
                    if hasattr(torch.cuda.get_device_properties(0), "total_mem")
                    else f"GPU memory - used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
                    f"total: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
                )

            # Set thread count
            faster_whisper_threads = 4
            if threads > 0:
                torch.set_num_threads(threads)
                faster_whisper_threads = threads

            # Load or reuse model
            if self._should_reload(model, device, device_index, compute_type, language, task):
                self.logger.debug(
                    "Loading model with config - model: %s, device: %s, compute_type: %s, "
                    "threads: %d, task: %s, language: %s",
                    model, device, compute_type, faster_whisper_threads, task, language,
                )

                # Clean up previous model if any
                if self.model is not None:
                    del self.model
                    self.model = None
                    gc.collect()
                    torch.cuda.empty_cache()

                self.model = load_model(
                    model,
                    device,
                    device_index=device_index,
                    compute_type=compute_type,
                    asr_options=asr_options,
                    vad_options=vad_options,
                    language=language,
                    task=task,
                    threads=faster_whisper_threads,
                )
                self._model_config = {
                    "model": model,
                    "device": device,
                    "device_index": device_index,
                    "compute_type": compute_type,
                    "language": language,
                    "task": task,
                }
                self.logger.debug("Transcription model loaded successfully")
            else:
                self.logger.debug("Reusing cached transcription model")

            # Transcribe
            result = self.model.transcribe(
                audio=audio, batch_size=batch_size, chunk_size=chunk_size, language=language
            )

            self.logger.debug("Completed transcription")
            return result  # type: ignore[no-any-return]

    def load_model(
        self,
        model_name: str,
        device: str,
        device_index: int,
        compute_type: str,
        asr_options: dict[str, Any],
        vad_options: dict[str, Any],
        language: str,
        task: str,
        threads: int,
    ) -> None:
        """
        Load WhisperX model.

        Args:
            model_name: Name/size of the model to load
            device: Device to load model on ('cpu' or 'cuda')
            device_index: Device index for multi-GPU setups
            compute_type: Computation precision
            asr_options: ASR model options
            vad_options: Voice Activity Detection options
            language: Target language
            task: Task type
            threads: Number of threads to use
        """
        self.logger.info(f"Loading model {model_name} on {device}")

        faster_whisper_threads = 4
        if threads > 0:
            torch.set_num_threads(threads)
            faster_whisper_threads = threads

        self.model = load_model(
            model_name,
            device,
            device_index=device_index,
            compute_type=compute_type,
            asr_options=asr_options,
            vad_options=vad_options,
            language=language,
            task=task,
            threads=faster_whisper_threads,
        )

    def unload_model(self) -> None:
        """Unload WhisperX model and free GPU memory."""
        if self.model:
            del self.model
            self.model = None
            self._model_config = None
            gc.collect()
            torch.cuda.empty_cache()
            self.logger.debug("Model unloaded and GPU memory cleared")
