from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import numpy as np
from whisperx import load_audio

from app.audio import split_stereo_to_mono
from app.callbacks import post_task_callback
from app.core.logging import logger
from app.domain.entities.task import Task as DomainTask
from app.domain.repositories.task_repository import ITaskRepository
from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.repositories.sqlalchemy_task_repository import (
    SQLAlchemyTaskRepository,
)
from app.schemas import (
    Metadata,
    Result,
    SpeechToTextProcessingParams,
    TaskStatus,
    TaskType,
)
from app.services.whisperx_wrapper_service import process_audio_common


def create_split_audio_tasks(
    parent_task_id: str,
    temp_file: str,
    filename: str,
    audio_duration: float,
    task_params: dict[str, Any],
    language: str | None,
    callback_url: str | None,
    repository: ITaskRepository,
) -> tuple[str, str]:
    """
    Create two child tasks for left and right audio channels.

    Args:
        parent_task_id: UUID of the parent task
        temp_file: Path to the temporary audio file
        filename: Original filename
        audio_duration: Duration of the audio
        task_params: Task processing parameters
        language: Language code
        callback_url: Callback URL (stored on parent, not children)
        repository: Task repository

    Returns:
        tuple[str, str]: UUIDs of left and right channel tasks
    """
    # Split the audio into left and right channels
    left_file, right_file = split_stereo_to_mono(temp_file)

    now = datetime.now(tz=timezone.utc)

    left_task = DomainTask(
        uuid=str(uuid4()),
        status=TaskStatus.processing,
        file_name=f"{filename}_left",
        audio_duration=audio_duration,
        language=language,
        task_type=TaskType.split_audio_channel,
        task_params={**task_params, "channel_file": left_file},
        parent_task_id=parent_task_id,
        channel="left",
        start_time=now,
    )

    right_task = DomainTask(
        uuid=str(uuid4()),
        status=TaskStatus.processing,
        file_name=f"{filename}_right",
        audio_duration=audio_duration,
        language=language,
        task_type=TaskType.split_audio_channel,
        task_params={**task_params, "channel_file": right_file},
        parent_task_id=parent_task_id,
        channel="right",
        start_time=now,
    )

    left_id = repository.add(left_task)
    right_id = repository.add(right_task)

    logger.info(
        "Created split audio child tasks: left=%s, right=%s for parent=%s",
        left_id,
        right_id,
        parent_task_id,
    )

    return left_id, right_id


def process_split_audio_channel(
    params: SpeechToTextProcessingParams,
    parent_task_id: str,
    channel: str,
) -> None:
    """
    Process a single audio channel and check if all sibling tasks are complete.

    Args:
        params: Speech-to-text processing parameters
        parent_task_id: UUID of the parent task
        channel: Channel identifier (left/right)
    """
    # Process the audio using the common processing function
    process_audio_common(params)

    # After processing, check if all child tasks are complete
    session = SessionLocal()
    repository: ITaskRepository = SQLAlchemyTaskRepository(session)

    try:
        _check_and_complete_parent_task(parent_task_id, repository)
    finally:
        session.close()


def _check_and_complete_parent_task(
    parent_task_id: str,
    repository: ITaskRepository,
) -> None:
    """
    Check if all child tasks are complete and update parent task accordingly.

    Args:
        parent_task_id: UUID of the parent task
        repository: Task repository
    """
    parent_task = repository.get_by_id(parent_task_id)
    if not parent_task:
        logger.error(f"Parent task not found: {parent_task_id}")
        return

    child_tasks = repository.get_by_parent_id(parent_task_id)

    if not child_tasks:
        logger.warning(f"No child tasks found for parent: {parent_task_id}")
        return

    # Check if all child tasks are complete (completed or failed)
    all_complete = all(
        task.status in [TaskStatus.completed, TaskStatus.failed] for task in child_tasks
    )

    if not all_complete:
        logger.debug(f"Not all child tasks complete for parent: {parent_task_id}")
        return

    # Check if any child task failed
    failed_tasks = [task for task in child_tasks if task.status == TaskStatus.failed]
    end_time = datetime.now(timezone.utc)

    if failed_tasks:
        # Parent task fails if any child fails
        error_messages = [
            f"{task.channel}: {task.error}" for task in failed_tasks if task.error
        ]
        repository.update(
            identifier=parent_task_id,
            update_data={
                "status": TaskStatus.failed,
                "error": "; ".join(error_messages),
                "end_time": end_time,
            },
        )
        logger.info(f"Parent task {parent_task_id} marked as failed")
    else:
        # All child tasks completed successfully - aggregate results
        aggregated_result = _aggregate_channel_results(child_tasks)

        # Calculate total duration from child tasks
        total_duration = sum(
            task.duration for task in child_tasks if task.duration is not None
        )

        repository.update(
            identifier=parent_task_id,
            update_data={
                "status": TaskStatus.completed,
                "result": aggregated_result,
                "duration": total_duration,
                "end_time": end_time,
            },
        )
        logger.info(f"Parent task {parent_task_id} marked as completed")

    if parent_task.callback_url:
        task = repository.get_by_id(parent_task_id)
        if task:
            metadata = Metadata(
                task_type=task.task_type,
                task_params=task.task_params,
                language=task.language,
                file_name=task.file_name,
                url=task.url,
                callback_url=task.callback_url,
                duration=task.duration,
                audio_duration=task.audio_duration,
                start_time=task.start_time,
                end_time=task.end_time,
            )
            result_payload = Result(
                status=task.status,
                result=task.result,
                metadata=metadata,
                error=task.error,
            )
            post_task_callback(parent_task.callback_url, result_payload.model_dump())


def _aggregate_channel_results(child_tasks: list[DomainTask]) -> dict[str, Any]:
    """
    Aggregate results from all child channel tasks.

    Args:
        child_tasks: List of completed child tasks

    Returns:
        dict[str, Any]: Aggregated results with channel information
    """
    results: dict[str, Any] = {"channels": {}}

    for task in child_tasks:
        if task.channel and task.result:
            results["channels"][task.channel] = task.result

    return results


def load_channel_audio(
    channel_file: str,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """
    Load audio from a channel file.

    Args:
        channel_file: Path to the channel audio file

    Returns:
        np.ndarray: Loaded audio data
    """
    return load_audio(channel_file)  # type: ignore[no-any-return]
