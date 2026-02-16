"""This module provides functions to filter aligned transcriptions."""
from typing import List

from app.schemas import AlignedTranscription, AlignmentSegment, LabeledSegment, LabeledWord

MAX_WORD_DURATION = 0.5

def filter_aligned_transcription(
    aligned_transcription: AlignedTranscription,
    speaker: str | None = None,
) -> AlignedTranscription:
    """
    Filter an AlignedTranscription instance by removing words with missing start, end, or score.
    Optionally applies a speaker label to the remaining words.
    """
    filtered_segments = []
    for segment in aligned_transcription.segments:
        filtered_words = [
            word
            for word in segment.words
            if all(
                [
                    word.start is not None,
                    word.end is not None,
                    word.score is not None,
                ]
            )
        ]
        if filtered_words:
            clamped_segment = clamp_segment_duration(segment, filtered_words, speaker)
            filtered_segments.append(clamped_segment)

    filtered_transcription = AlignedTranscription(
        segments=filtered_segments,
        word_segments=[]
    )
    return filtered_transcription


def clamp_segment_duration(
    aligned_segment: AlignmentSegment,
    filtered_words: List[LabeledWord],
    speaker: str | None = None,
) -> LabeledSegment:
    first_word = filtered_words[0]
    last_word = filtered_words[-1]

    start = (
        first_word.end - MAX_WORD_DURATION
        if (first_word.end - first_word.start) > MAX_WORD_DURATION
        else first_word.start
    )

    end = (
        last_word.start + MAX_WORD_DURATION
        if (last_word.end - last_word.start) > MAX_WORD_DURATION
        else last_word.end
    )

    if speaker:
        for word in filtered_words:
            word.speaker = speaker

    return LabeledSegment(
        start=start,
        end=end,
        text=aligned_segment.text,
        words=filtered_words,
        speaker=speaker,
    )