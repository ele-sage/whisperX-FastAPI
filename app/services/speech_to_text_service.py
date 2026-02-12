import json
from datetime import datetime, timezone
from typing import Any, List
from uuid import uuid4
from fastapi import BackgroundTasks
from whisperx import load_audio

from app.audio import (
    get_audio_duration_from_file,
    process_audio_file,
    split_stereo_to_mono, is_stereo_audio,
)
from app.callbacks import send_task_result_callback
from app.core.logging import logger
from app.domain.entities.task import Task as DomainTask
from app.domain.repositories.task_repository import ITaskRepository
from app.files import safe_remove_file
from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.repositories.sqlalchemy_task_repository import (
    SQLAlchemyTaskRepository,
)
from app.schemas import (
    ProcessingConfig,
    Response,
    SpeechToTextProcessingParams,
    TaskStatus,
    TaskType, AlignmentSegment, LabeledSegment, AlignedTranscription, LabeledWord,
)
from app.services.whisperx_wrapper_service import process_audio_common

def _create_task(
    filename: str,
    audio_duration: float,
    task_params: dict[str, Any],
    task_type: str,
    language: str | None = None,
    url: str | None = None,
    callback_url: str | None = None,
    parent_task_id: str | None = None,
    channel: str | None = None,
) -> DomainTask:
    """Helper to create a DomainTask with common defaults."""
    return DomainTask(
        uuid=str(uuid4()),
        status=TaskStatus.processing,
        file_name=filename,
        audio_duration=audio_duration,
        language=language,
        task_type=task_type,
        task_params=task_params,
        url=url,
        callback_url=callback_url,
        parent_task_id=parent_task_id,
        channel=channel,
        start_time=datetime.now(tz=timezone.utc),
    )

def process_speech_to_text(
    temp_file: str,
    filename: str,
    background_tasks: BackgroundTasks,
    config: ProcessingConfig,
    repository: ITaskRepository,
    split_audio: bool,
    callback_url: str | None = None,
    url: str | None = None,
) -> Response:
    """Process a speech-to-text request for an audio file."""
    try:
        audio_duration = get_audio_duration_from_file(temp_file)
        logger.info("Audio file %s length: %s seconds", filename, audio_duration)

        task_params = {
            **config.model_params.model_dump(),
            **config.align_params.model_dump(),
            "asr_options": config.asr_options.model_dump(),
            "vad_options": config.vad_options.model_dump(),
            **config.diarize_params.model_dump(),
        }

        if split_audio and is_stereo_audio(temp_file):
            task_params.update({"min_speakers": 1, "max_speakers": 1})

            task = _create_task(
                filename=filename,
                audio_duration=audio_duration,
                task_params=task_params,
                task_type=TaskType.split_audio_parent,
                language=config.model_params.language,
                url=url,
                callback_url=callback_url,
            )
            task_id = repository.add(task)
            logger.info("Split audio parent task added: ID %s", task_id)

            return _process_split_audio(
                parent_id=task_id,
                temp_file=temp_file,
                filename=filename,
                audio_duration=audio_duration,
                task_params=task_params,
                background_tasks=background_tasks,
                config=config,
                repository=repository,
            )

        task = _create_task(
            filename=filename,
            audio_duration=audio_duration,
            task_params=task_params,
            task_type=TaskType.full_process,
            language=config.model_params.language,
            url=url,
            callback_url=callback_url,
        )
        task_id = repository.add(task)
        logger.info("Task added to database: ID %s", task_id)

        audio_params = SpeechToTextProcessingParams(
            audio=process_audio_file(temp_file),
            identifier=task_id,
            vad_options=config.vad_options,
            asr_options=config.asr_options,
            whisper_model_params=config.model_params,
            alignment_params=config.align_params,
            diarization_params=config.diarize_params,
            callback_url=callback_url,
        )
        background_tasks.add_task(process_audio_common, audio_params)
        logger.info("Background task scheduled: ID %s", task_id)

        return Response(identifier=task_id, message="Task queued")
    finally:
        safe_remove_file(temp_file)

def _process_split_audio(
    parent_id: str,
    temp_file: str,
    filename: str,
    audio_duration: float,
    task_params: dict[str, Any],
    background_tasks: BackgroundTasks,
    config: ProcessingConfig,
    repository: ITaskRepository,
) -> Response:
    """
    Handle split audio processing mode.

    Creates a parent task and two child tasks for left/right channels,
    then schedules background processing for each channel.
    """
    left_file, right_file = split_stereo_to_mono(temp_file)

    for channel, channel_file in [("left", left_file), ("right", right_file)]:
        channel_task_params = {**task_params, "channel_file": channel_file}
        child_task = _create_task(
            filename=f"{filename}_{channel}",
            audio_duration=audio_duration,
            task_params=channel_task_params,
            task_type=TaskType.split_audio_channel,
            language=config.model_params.language,
            parent_task_id=parent_id,
            channel=channel,
        )
        channel_id = repository.add(child_task)
        logger.info(
            "Created split audio child task: %s channel=%s for parent=%s",
            channel_id,
            channel,
            parent_id,
        )
        try:
            audio = load_audio(child_task.task_params["channel_file"])
        finally:
            safe_remove_file(child_task.task_params["channel_file"])
        params = SpeechToTextProcessingParams(
            audio=audio,
            identifier=channel_id,
            vad_options=config.vad_options,
            asr_options=config.asr_options,
            whisper_model_params=config.model_params,
            alignment_params=config.align_params,
            diarization_params=config.diarize_params,
        )
        background_tasks.add_task(_process_split_audio_channel, params, parent_id, channel)

    logger.info(
        "Background tasks scheduled for split audio processing: parent=%s",
        parent_id,
    )

    return Response(identifier=parent_id, message="Split audio task queued")


def _process_split_audio_channel(
    params: SpeechToTextProcessingParams,
    parent_task_id: str,
    channel: str,
) -> None:
    """
    Process a single audio channel and check if all sibling tasks are complete.
    """
    process_audio_common(params, channel_name=channel)

    session = SessionLocal()
    repository: ITaskRepository = SQLAlchemyTaskRepository(session)

    try:
        _check_and_complete_parent_task(parent_task_id, repository)
    finally:
        session.close()


def _merge_channel_results(
    channels_result: dict[str, AlignedTranscription],
) -> AlignedTranscription:
    """
    Merge transcription results from multiple channels into a single aligned transcription.

    This function:
    1. Iterates through each channel's transcription.
    2. Converts segments and words into LabeledSegment/LabeledWord objects.
    3. Assigns the channel name as the 'speaker'.
    4. Collects all segments and words into flat lists.
    5. Sorts the final lists chronologically by start time.
    """
    merged_segments: List[LabeledSegment] = []
    merged_words: List[LabeledWord] = []

    for channel, transcription in channels_result.items():
        for segment in transcription.segments:
            # Create a new LabeledSegment, carrying over all data + the speaker
            # Note: We use model_dump to safely copy fields from the existing segment
            new_segment = LabeledSegment(
                **segment.model_dump(exclude={"speaker", "words"}),
                # exclude potentially existing or incompatible fields
                words=[],  # We will populate words separately if needed, but usually segments contain them
                speaker=channel
            )

            # If the original segment had words, we must upgrade them to LabeledWord as well
            # so they match the schema of LabeledSegment.words
            if segment.words:
                new_segment_words = []
                for word in segment.words:
                    labeled_word = LabeledWord(
                        **word.model_dump(exclude={"speaker"}),
                        speaker=channel
                    )
                    new_segment_words.append(labeled_word)
                new_segment.words = new_segment_words

            merged_segments.append(new_segment)

        # --- Process Word Segments (Standalone Words) ---
        # If your transcription object has a separate list of words outside of segments
        if transcription.word_segments:
            for word in transcription.word_segments:
                labeled_word = LabeledWord(
                    **word.model_dump(exclude={"speaker"}),
                    speaker=channel
                )
                merged_words.append(labeled_word)

    # --- Sort Chronologically ---
    # Sort segments by start time.
    # We use a default of 0.0 for start time to prevent crashes if 'start' is None (though it shouldn't be).
    merged_segments.sort(key=lambda s: s.start or 0.0)

    # Sort word segments by start time.
    merged_words.sort(key=lambda w: w.start or 0.0)

    return AlignedTranscription(
        segments=merged_segments,
        word_segments=merged_words
    )

def _check_and_complete_parent_task(
    parent_task_id: str,
    repository: ITaskRepository,
) -> None:
    """Check if all child tasks are complete and update parent task accordingly."""
    parent_task = repository.get_by_id(parent_task_id)
    child_tasks = repository.get_by_parent_id(parent_task_id)

    if not parent_task or not child_tasks:
        logger.warning(f"Parent task {parent_task_id} or child tasks not found/empty")
        return

    if not all(t.status in [TaskStatus.completed, TaskStatus.failed] for t in child_tasks):
        return

    end_time = datetime.now(timezone.utc)
    failed_tasks = [t for t in child_tasks if t.status == TaskStatus.failed]

    if failed_tasks:
        error_msg = "; ".join(f"{t.channel}: {t.error}" for t in failed_tasks if t.error)
        update_data = {
            "status": TaskStatus.failed,
            "error": error_msg,
            "end_time": end_time,
        }
    else:
        channels: dict[str, AlignedTranscription] = {}
        for task in child_tasks:
            if not task.channel or not task.result:
                continue
            if isinstance(task.result, dict):
                transcription_data = AlignedTranscription.model_validate(task.result)
            else:
                transcription_data = task.result

            channels[task.channel] = transcription_data

        results = _merge_channel_results(channels)
        total_duration = sum(t.duration for t in child_tasks if t.duration)
        update_data = {
            "status": TaskStatus.completed,
            "result": results.model_dump(),
            "duration": total_duration,
            "end_time": end_time,
        }

    repository.update(identifier=parent_task_id, update_data=update_data)
    logger.info(f"Parent task {parent_task_id} marked as {update_data['status']}")

    if parent_task.callback_url:
        if task := repository.get_by_id(parent_task_id):
            send_task_result_callback(task)
