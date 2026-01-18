from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks

from app.audio import get_audio_duration, process_audio_file
from app.core.logging import logger
from app.domain.entities.task import Task as DomainTask
from app.domain.repositories.task_repository import ITaskRepository
from app.schemas import (
    AlignmentParams,
    ASROptions,
    DiarizationParams,
    Response,
    SpeechToTextProcessingParams,
    TaskStatus,
    TaskType,
    VADOptions,
    WhisperModelParams,
)
from app.services.split_audio_service import (
    create_split_audio_tasks,
    load_channel_audio,
    process_split_audio_channel,
)
from app.services.whisperx_wrapper_service import process_audio_common


def build_task_params(
    model_params: WhisperModelParams,
    align_params: AlignmentParams,
    asr_options: ASROptions,
    vad_options: VADOptions,
    diarize_params: DiarizationParams,
) -> dict[str, Any]:
    """
    Build task parameters dictionary from various parameter objects.

    Args:
        model_params: Whisper model parameters.
        align_params: Alignment parameters.
        asr_options: ASR options parameters.
        vad_options: VAD options parameters.
        diarize_params: Diarization parameters.

    Returns:
        dict[str, Any]: Combined task parameters.
    """
    return {
        **model_params.model_dump(),
        **align_params.model_dump(),
        "asr_options": asr_options.model_dump(),
        "vad_options": vad_options.model_dump(),
        **diarize_params.model_dump(),
    }


def process_speech_to_text(
    temp_file: str,
    filename: str,
    background_tasks: BackgroundTasks,
    model_params: WhisperModelParams,
    align_params: AlignmentParams,
    diarize_params: DiarizationParams,
    asr_options: ASROptions,
    vad_options: VADOptions,
    repository: ITaskRepository,
    split_audio: bool,
    callback_url: str | None = None,
    url: str | None = None,
) -> Response:
    """
    Process a speech-to-text request for an audio file.

    This function handles both standard and split audio processing modes,
    creating appropriate tasks and scheduling background processing.

    Args:
        temp_file: Path to the temporary audio file.
        filename: Original filename.
        background_tasks: FastAPI background tasks handler.
        model_params: Whisper model parameters.
        align_params: Alignment parameters.
        diarize_params: Diarization parameters.
        asr_options: ASR options parameters.
        vad_options: VAD options parameters.
        repository: Task repository dependency.
        split_audio: If True, split stereo audio and process channels separately.
        callback_url: Optional URL to notify when the task is done.
        url: Optional source URL (for URL-based requests).

    Returns:
        Response: Confirmation message with task identifier.
    """
    audio = process_audio_file(temp_file)
    audio_duration = get_audio_duration(audio)
    logger.info("Audio file %s length: %s seconds", filename, audio_duration)

    task_params = build_task_params(
        model_params=model_params,
        align_params=align_params,
        asr_options=asr_options,
        vad_options=vad_options,
        diarize_params=diarize_params,
    )

    if split_audio:
        task_params.update({"min_speakers": 1, "max_speakers": 1})

        return _process_split_audio(
            temp_file=temp_file,
            filename=filename,
            audio_duration=audio_duration,
            task_params=task_params,
            background_tasks=background_tasks,
            model_params=model_params,
            align_params=align_params,
            diarize_params=diarize_params,
            asr_options=asr_options,
            vad_options=vad_options,
            repository=repository,
            callback_url=callback_url,
            url=url,
        )

    return _process_standard_audio(
        audio=audio,
        filename=filename,
        audio_duration=audio_duration,
        task_params=task_params,
        background_tasks=background_tasks,
        model_params=model_params,
        align_params=align_params,
        diarize_params=diarize_params,
        asr_options=asr_options,
        vad_options=vad_options,
        repository=repository,
        callback_url=callback_url,
        url=url,
    )


def _process_split_audio(
    temp_file: str,
    filename: str,
    audio_duration: float,
    task_params: dict[str, Any],
    background_tasks: BackgroundTasks,
    model_params: WhisperModelParams,
    align_params: AlignmentParams,
    diarize_params: DiarizationParams,
    asr_options: ASROptions,
    vad_options: VADOptions,
    repository: ITaskRepository,
    callback_url: str | None = None,
    url: str | None = None,
) -> Response:
    """
    Handle split audio processing mode.

    Creates a parent task and two child tasks for left/right channels,
    then schedules background processing for each channel.

    Args:
        temp_file: Path to the temporary audio file.
        filename: Original filename.
        audio_duration: Duration of the audio in seconds.
        task_params: Combined task parameters.
        background_tasks: FastAPI background tasks handler.
        model_params: Whisper model parameters.
        align_params: Alignment parameters.
        diarize_params: Diarization parameters.
        asr_options: ASR options parameters.
        vad_options: VAD options parameters.
        repository: Task repository dependency.
        callback_url: Optional URL to notify when the task is done.
        url: Optional source URL (for URL-based requests).

    Returns:
        Response: Confirmation message with parent task identifier.
    """
    parent_task = DomainTask(
        uuid=str(uuid4()),
        status=TaskStatus.processing,
        file_name=filename,
        audio_duration=audio_duration,
        language=model_params.language,
        task_type=TaskType.split_audio_parent,
        task_params=task_params,
        url=url,
        callback_url=callback_url,
        start_time=datetime.now(tz=timezone.utc),
    )

    parent_id = repository.add(parent_task)
    logger.info("Parent split audio task added to database: ID %s", parent_id)

    left_id, right_id = create_split_audio_tasks(
        parent_task_id=parent_id,
        temp_file=temp_file,
        filename=filename,
        audio_duration=audio_duration,
        task_params=task_params,
        language=model_params.language,
        callback_url=callback_url,
        repository=repository,
    )

    _schedule_channel_task(
        channel_id=left_id,
        parent_id=parent_id,
        channel="left",
        repository=repository,
        background_tasks=background_tasks,
        model_params=model_params,
        align_params=align_params,
        diarize_params=diarize_params,
        asr_options=asr_options,
        vad_options=vad_options,
    )

    _schedule_channel_task(
        channel_id=right_id,
        parent_id=parent_id,
        channel="right",
        repository=repository,
        background_tasks=background_tasks,
        model_params=model_params,
        align_params=align_params,
        diarize_params=diarize_params,
        asr_options=asr_options,
        vad_options=vad_options,
    )

    logger.info(
        "Background tasks scheduled for split audio processing: parent=%s",
        parent_id,
    )

    return Response(identifier=parent_id, message="Split audio task queued")


def _schedule_channel_task(
    channel_id: str,
    parent_id: str,
    channel: str,
    repository: ITaskRepository,
    background_tasks: BackgroundTasks,
    model_params: WhisperModelParams,
    align_params: AlignmentParams,
    diarize_params: DiarizationParams,
    asr_options: ASROptions,
    vad_options: VADOptions,
) -> None:
    """
    Schedule a background task for processing a single audio channel.

    Args:
        channel_id: UUID of the channel task.
        parent_id: UUID of the parent task.
        channel: Channel identifier (left/right).
        repository: Task repository dependency.
        background_tasks: FastAPI background tasks handler.
        model_params: Whisper model parameters.
        align_params: Alignment parameters.
        diarize_params: Diarization parameters.
        asr_options: ASR options parameters.
        vad_options: VAD options parameters.
    """
    task = repository.get_by_id(channel_id)
    if task and task.task_params:
        audio = load_channel_audio(task.task_params["channel_file"])
        params = SpeechToTextProcessingParams(
            audio=audio,
            identifier=channel_id,
            vad_options=vad_options,
            asr_options=asr_options,
            whisper_model_params=model_params,
            alignment_params=align_params,
            diarization_params=diarize_params,
        )
        background_tasks.add_task(
            process_split_audio_channel, params, parent_id, channel
        )


def _process_standard_audio(
    audio: Any,
    filename: str,
    audio_duration: float,
    task_params: dict[str, Any],
    background_tasks: BackgroundTasks,
    model_params: WhisperModelParams,
    align_params: AlignmentParams,
    diarize_params: DiarizationParams,
    asr_options: ASROptions,
    vad_options: VADOptions,
    repository: ITaskRepository,
    callback_url: str | None = None,
    url: str | None = None,
) -> Response:
    """
    Handle standard (non-split) audio processing mode.

    Creates a single task and schedules background processing.

    Args:
        audio: Processed audio data.
        filename: Original filename.
        audio_duration: Duration of the audio in seconds.
        task_params: Combined task parameters.
        background_tasks: FastAPI background tasks handler.
        model_params: Whisper model parameters.
        align_params: Alignment parameters.
        diarize_params: Diarization parameters.
        asr_options: ASR options parameters.
        vad_options: VAD options parameters.
        repository: Task repository dependency.
        callback_url: Optional URL to notify when the task is done.
        url: Optional source URL (for URL-based requests).

    Returns:
        Response: Confirmation message with task identifier.
    """
    task = DomainTask(
        uuid=str(uuid4()),
        status=TaskStatus.processing,
        file_name=filename,
        audio_duration=audio_duration,
        language=model_params.language,
        task_type=TaskType.full_process,
        task_params=task_params,
        url=url,
        callback_url=callback_url,
        start_time=datetime.now(tz=timezone.utc),
    )

    identifier = repository.add(task)
    logger.info("Task added to database: ID %s", identifier)

    audio_params = SpeechToTextProcessingParams(
        audio=audio,
        identifier=identifier,
        vad_options=vad_options,
        asr_options=asr_options,
        whisper_model_params=model_params,
        alignment_params=align_params,
        diarization_params=diarize_params,
        callback_url=callback_url
    )

    background_tasks.add_task(process_audio_common, audio_params)
    logger.info("Background task scheduled for processing: ID %s", identifier)

    return Response(identifier=identifier, message="Task queued")

