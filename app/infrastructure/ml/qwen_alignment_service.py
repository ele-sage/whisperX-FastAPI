"""Qwen3ForcedAligner implementation of alignment service."""

import gc
import nltk
from typing import Any, List

import numpy as np
import torch
from qwen_asr import Qwen3ForcedAligner
from whisperx.audio import SAMPLE_RATE

from app.core.gpu import gpu_lock
from app.core.logging import logger

BATCH_SIZE = 16

class Qwen3AlignmentService:
    """
    Qwen3-based implementation of alignment service.
    Includes sub-segmentation logic similar to WhisperX.
    """

    def __init__(self) -> None:
        """Initialize the alignment service."""
        self.model: Any = None
        self.logger = logger

        # Ensure NLTK data is available for sentence splitting
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

    def align(
        self,
        transcript: list[dict[str, Any]],
        audio: np.ndarray,
        language_code: str,
        device: str,
        align_model: str | None = None,
        interpolate_method: str = "nearest",
        return_char_alignments: bool = False,
    ) -> dict[str, Any]:
        """
        Align transcript to audio using Qwen3, then refine segments by sentence.
        """
        self.logger.debug(f"Starting alignment for {language_code} on {device}")

        with gpu_lock("alignment"):
            return self._align_inner(
                transcript, audio, language_code, device,
                align_model, interpolate_method, return_char_alignments,
            )

    def _align_inner(
        self,
        transcript: list[dict[str, Any]],
        audio: np.ndarray,
        language_code: str,
        device: str,
        align_model: str | None,
        interpolate_method: str,
        return_char_alignments: bool,
    ) -> dict[str, Any]:
        """Inner alignment logic, called while holding the GPU semaphore."""
        if self.model is None:
            self.load_model(language_code, device)

        qwen_lang = "French" if language_code == "fr" else "English"

        final_segments = []
        all_word_segments = []

        total_segments = len(transcript)

        # --- BATCH PROCESSING LOOP ---
        for i in range(0, total_segments, BATCH_SIZE):
            batch_transcript = transcript[i : i + BATCH_SIZE]

            # Prepare inputs
            audio_batch = []
            text_batch = []
            offsets = []
            valid_indices = []

            for idx, segment in enumerate(batch_transcript):
                text = segment['text'].strip()
                if not text:
                    continue

                # 1. Slice Audio
                start_sample = int(segment['start'] * SAMPLE_RATE)
                end_sample = int(segment['end'] * SAMPLE_RATE)

                if start_sample >= len(audio): continue
                end_sample = min(end_sample, len(audio))

                audio_slice = audio[start_sample:end_sample]
                if len(audio_slice) < 320: continue

                audio_batch.append((audio_slice, SAMPLE_RATE))
                text_batch.append(text)
                offsets.append(segment['start'])
                valid_indices.append(idx)

            if not audio_batch:
                continue

            try:
                # 2. Run Qwen Alignment
                results = self.model.align(
                    audio=audio_batch,
                    text=text_batch,
                    language=qwen_lang,
                )
            except Exception as e:
                self.logger.error(f"Qwen3 Alignment batch failed: {e}")
                final_segments.extend(batch_transcript)
                continue

            # 3. Process Results
            for j, result in enumerate(results):
                original_segment = batch_transcript[valid_indices[j]]
                time_offset = offsets[j]

                # Convert Qwen items to Dictionary Words with absolute timing
                aligned_words = []
                for item in result:
                    word_dict = {
                        "word": item.text,
                        "start": round(item.start_time + time_offset, 3),
                        "end": round(item.end_time + time_offset, 3),
                        "score": 1.0
                    }
                    aligned_words.append(word_dict)
                    all_word_segments.append(word_dict)

                # 4. SUB-SEGMENTATION LOGIC (Refining the segment)
                # Instead of just attaching words to the big segment, we split it.
                if aligned_words:
                    refined_subsegments = self._refine_segments_by_sentence(
                        original_text=original_segment['text'],
                        aligned_words=aligned_words
                    )
                    final_segments.extend(refined_subsegments)
                else:
                    # Fallback if no words found (e.g. silence or "...")
                    # We must create a copy and explicitly add an empty words list
                    # to satisfy the AlignedTranscription schema.
                    fallback_segment = original_segment.copy()
                    fallback_segment["words"] = []
                    final_segments.append(fallback_segment)

        return {
            "segments": final_segments,
            "word_segments": all_word_segments
        }

    def _refine_segments_by_sentence(
        self,
        original_text: str,
        aligned_words: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Splits a segment into smaller segments based on sentence boundaries,
        assigning the correct aligned words to each sentence.
        """
        # 1. Split text into sentences using NLTK
        sentences = nltk.sent_tokenize(original_text)

        new_segments = []
        word_cursor = 0
        total_words = len(aligned_words)

        for sentence in sentences:
            sentence_clean = sentence.strip()
            if not sentence_clean:
                continue

            # 2. Approximate how many words belong to this sentence
            # We split by space to count words. Ideally, we would match word-for-word,
            # but counting is usually sufficient for alignment redistribution.
            # (Qwen and NLTK usually agree on word counts for standard languages)
            sentence_word_count = len(sentence_clean.split())

            # 3. Gather the words for this sentence
            segment_words = []

            # Consume words from the aligned list
            # We try to consume 'sentence_word_count', but we stop if we hit the end
            target_index = min(word_cursor + sentence_word_count, total_words)

            # Simple heuristic: Just take the next N words.
            # A more robust solution would fuzzy match the word text,
            # but this mirrors the basic WhisperX logic.
            current_batch = aligned_words[word_cursor:target_index]
            segment_words.extend(current_batch)

            word_cursor = target_index

            # 4. Create the new segment
            if segment_words:
                start_time = segment_words[0]['start']
                end_time = segment_words[-1]['end']

                new_segments.append({
                    "text": sentence_clean,
                    "start": start_time,
                    "end": end_time,
                    "words": segment_words
                })

        # If any words remain (due to mismatch in splitting), attach to last segment
        if word_cursor < total_words and new_segments:
            remaining = aligned_words[word_cursor:]
            new_segments[-1]['words'].extend(remaining)
            new_segments[-1]['end'] = remaining[-1]['end']
            # Optionally append text to last segment, but usually not needed for timeline

        # If NLTK failed to return anything but we have words, return 1 segment
        if not new_segments and aligned_words:
             new_segments.append({
                "text": original_text,
                "start": aligned_words[0]['start'],
                "end": aligned_words[-1]['end'],
                "words": aligned_words
            })

        return new_segments

    def load_model(
        self, language_code: str, device: str, model_name: str | None = None
    ) -> None:
        """Load alignment model."""
        self.logger.info(f"Loading Qwen3 model on {device}...")
        dtype = torch.bfloat16 if device == 'cuda' else torch.int8
        device_map = "cuda:0" if device == 'cuda' else "cpu"
        self.model = Qwen3ForcedAligner.from_pretrained(
            "Qwen/Qwen3-ForcedAligner-0.6B",
            dtype=dtype,
            device_map=device_map,
        )

    def unload_model(self) -> None:
        if self.model:
            del self.model
            self.model = None
        gc.collect()
        torch.cuda.empty_cache()