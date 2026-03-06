"""WhisperX implementation of alignment service."""

import gc
from typing import Any

import numpy as np
import torch
from whisperx import align, load_align_model

from app.core.gpu import gpu_lock
from app.core.logging import logger


class WhisperXAlignmentService:
    """
    WhisperX-based implementation of alignment service.

    The alignment model is lazily loaded on first call and cached for reuse.
    Since the model is language-specific, it is reloaded when the language changes.
    GPU access is serialized via a semaphore so concurrent requests queue safely.
    """

    def __init__(self) -> None:
        """Initialize the alignment service."""
        self.model: Any = None
        self.metadata: Any = None
        self._model_language: str | None = None
        self._model_device: str | None = None
        self._vram_usage_bytes: float = 0.0
        self.logger = logger

    def get_vram_usage(self) -> float:
        """Get the estimated VRAM usage by this specific model instance in MB."""
        return self._vram_usage_bytes / (1024 * 1024)

    def align(
        self,
        transcript: list[dict[str, Any]],
        audio: np.ndarray[Any, np.dtype[np.float32]],
        language_code: str,
        device: str,
        align_model: str | None = None,
        interpolate_method: str = "nearest",
        return_char_alignments: bool = False,
    ) -> dict[str, Any]:
        """
        Align transcript to audio using WhisperX alignment.

        The model is loaded once per language and reused.
        GPU access is serialized.
        """
        with gpu_lock("alignment"):
            self.logger.debug(
                "Starting alignment for language code: %s on device: %s",
                language_code,
                device,
            )

            # Log GPU memory
            if torch.cuda.is_available():
                self.logger.debug(
                    f"GPU memory - used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
                    f"available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB"
                )

            # Load or reuse model (reload if language or device changed)
            if (
                self.model is None
                or self._model_language != language_code
                or self._model_device != device
            ):
                self.logger.debug(
                    "Loading align model - language_code: %s, device: %s",
                    language_code,
                    device,
                )

                # Clean up previous model if any
                if self.model is not None:
                    self.model = None
                    self.metadata = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    self._vram_usage_bytes = 0.0

                vram_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0.0

                self.model, self.metadata = load_align_model(
                    language_code=language_code, device=device, model_name=align_model
                )
                self._model_language = language_code
                self._model_device = device
                
                vram_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0.0
                self._vram_usage_bytes = max(0.0, vram_after - vram_before)
                self.logger.debug(f"Alignment model loaded. VRAM footprint: {self.get_vram_usage():.2f} MB")
            else:
                self.logger.debug("Reusing cached alignment model")

            # Perform alignment
            result = align(
                transcript,
                self.model,
                self.metadata,
                audio,
                device,
                interpolate_method=interpolate_method,
                return_char_alignments=return_char_alignments,
            )

            self.logger.debug("Completed alignment")
            return result  # type: ignore[no-any-return]

    def load_model(
        self, language_code: str, device: str, model_name: str | None = None
    ) -> None:
        """
        Load alignment model for a specific language.

        Args:
            language_code: Language code for the alignment model
            device: Device to load model on ('cpu' or 'cuda')
            model_name: Specific model name to use (optional)
        """
        self.logger.info(f"Loading alignment model for {language_code} on {device}")
        vram_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0.0
        
        self.model, self.metadata = load_align_model(
            language_code=language_code, device=device, model_name=model_name
        )
        self._model_language = language_code
        self._model_device = device
        
        vram_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0.0
        self._vram_usage_bytes = max(0.0, vram_after - vram_before)

    def unload_model(self) -> None:
        """Unload alignment model and free GPU memory."""
        if self.model:
            self.model = None
        if self.metadata:
            self.metadata = None
        self._model_language = None
        self._model_device = None
        self._vram_usage_bytes = 0.0
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.debug("Alignment model unloaded and GPU memory cleared")
