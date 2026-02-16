"""WhisperX implementation of diarization service."""

import gc
from typing import Any

import numpy as np
import pandas as pd
import torch
from whisperx.diarize import DiarizationPipeline

from app.core.gpu import gpu_lock
from app.core.logging import logger


class WhisperXDiarizationService:
    """
    WhisperX/PyAnnote-based implementation of diarization service.

    The diarization model is lazily loaded on first call and cached for reuse.
    GPU access is serialized via a semaphore so concurrent requests queue safely.
    """

    def __init__(self, hf_token: str) -> None:
        """
        Initialize the diarization service.

        Args:
            hf_token: HuggingFace authentication token for model access
        """
        self.hf_token = hf_token
        self.model: Any = None
        self._model_device: str | None = None
        self.logger = logger

    def diarize(
        self,
        audio: np.ndarray[Any, np.dtype[np.float32]],
        device: str,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> pd.DataFrame:
        """
        Identify speakers using PyAnnote diarization model.

        The model is loaded once and reused across calls.
        GPU access is serialized.
        """
        with gpu_lock("diarization"):
            self.logger.debug("Starting diarization with device: %s", device)

            # Log GPU memory
            if torch.cuda.is_available():
                self.logger.debug(
                    f"GPU memory - used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
                    f"available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
                )

            # Load or reuse model
            if self.model is None or self._model_device != device:
                self.logger.debug("Loading diarization model on %s", device)

                # Clean up previous model if device changed
                if self.model is not None:
                    del self.model
                    self.model = None
                    gc.collect()
                    torch.cuda.empty_cache()

                self.model = DiarizationPipeline(
                    use_auth_token=self.hf_token, device=device
                )
                self._model_device = device
                self.logger.debug("Diarization model loaded successfully")
            else:
                self.logger.debug("Reusing cached diarization model")

            # Perform diarization
            result = self.model(
                audio=audio, min_speakers=min_speakers, max_speakers=max_speakers
            )

            self.logger.debug("Completed diarization with device: %s", device)
            return result  # type: ignore[no-any-return]

    def load_model(self, device: str, hf_token: str) -> None:
        """
        Load diarization model.

        Args:
            device: Device to load model on ('cpu' or 'cuda')
            hf_token: HuggingFace authentication token
        """
        self.logger.info(f"Loading diarization model on {device}")
        self.hf_token = hf_token
        self.model = DiarizationPipeline(use_auth_token=self.hf_token, device=device)
        self._model_device = device

    def unload_model(self) -> None:
        """Unload diarization model and free GPU memory."""
        if self.model:
            del self.model
            self.model = None
            self._model_device = None
            gc.collect()
            torch.cuda.empty_cache()
            self.logger.debug("Diarization model unloaded and GPU memory cleared")
