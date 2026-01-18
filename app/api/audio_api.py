"""
This module contains the FastAPI routes for speech-to-text processing.

It includes endpoints for processing uploaded audio files and audio files from URLs.
"""

import logging
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Query,
    UploadFile,
)

from app.api.dependencies import get_file_service, get_task_repository
from app.core.exceptions import FileValidationError
from app.core.logging import logger
from app.domain.repositories.task_repository import ITaskRepository
from app.files import ALLOWED_EXTENSIONS
from app.schemas import (
    AlignmentParams,
    ASROptions,
    DiarizationParams,
    Response,
    VADOptions,
    WhisperModelParams,
)
from app.services.file_service import FileService

from app.api.callbacks import task_callback_router
from app.callbacks import validate_callback_url_dependency
from app.services.speech_to_text_service import process_speech_to_text


# Configure logging
logging.basicConfig(level=logging.INFO)

stt_router = APIRouter()


@stt_router.post("/speech-to-text", tags=["Speech-2-Text"])
async def speech_to_text(
    background_tasks: BackgroundTasks,
    model_params: WhisperModelParams = Depends(),
    align_params: AlignmentParams = Depends(),
    diarize_params: DiarizationParams = Depends(),
    asr_options_params: ASROptions = Depends(),
    vad_options_params: VADOptions = Depends(),
    file: UploadFile = File(...),
    callback_url: str | None = Depends(validate_callback_url_dependency),
    split_audio: bool = Query(
        default=False,
        description="Split stereo audio into separate channels for individual processing",
    ),
    repository: ITaskRepository = Depends(get_task_repository),
    file_service: FileService = Depends(get_file_service),
) -> Response:
    """
    Process an uploaded audio file for speech-to-text conversion.

    Args:
        background_tasks (BackgroundTasks): Background tasks dependency.
        model_params (WhisperModelParams): Whisper model parameters.
        align_params (AlignmentParams): Alignment parameters.
        diarize_params (DiarizationParams): Diarization parameters.
        asr_options_params (ASROptions): ASR options parameters.
        vad_options_params (VADOptions): VAD options parameters.
        file (UploadFile): Uploaded audio file.
        callback_url (str | None): Optional URL to call back when processing is complete.
        split_audio (bool): If True, split stereo audio and process channels separately.
        repository (ITaskRepository): Task repository dependency.
        file_service (FileService): File service dependency.

    Returns:
        Response: Confirmation message of task queuing.
    """
    logger.info("Received file upload request: %s", file.filename)

    # Validate file using file service
    if file.filename is None:
        raise FileValidationError(filename="unknown", reason="Filename is missing")

    file_service.validate_file_extension(file.filename, ALLOWED_EXTENSIONS)

    # Save file using file service
    temp_file = file_service.save_upload(file)
    logger.info("%s saved as temporary file: %s", file.filename, temp_file)

    return process_speech_to_text(
        temp_file=temp_file,
        filename=file.filename,
        background_tasks=background_tasks,
        model_params=model_params,
        align_params=align_params,
        diarize_params=diarize_params,
        asr_options=asr_options_params,
        vad_options=vad_options_params,
        repository=repository,
        split_audio=split_audio,
        callback_url=callback_url,
    )


@stt_router.post(
    "/speech-to-text-url", callbacks=task_callback_router.routes, tags=["Speech-2-Text"]
)
async def speech_to_text_url(
    background_tasks: BackgroundTasks,
    model_params: WhisperModelParams = Depends(),
    align_params: AlignmentParams = Depends(),
    diarize_params: DiarizationParams = Depends(),
    asr_options_params: ASROptions = Depends(),
    vad_options_params: VADOptions = Depends(),
    url: str = Form(...),
    callback_url: str | None = Depends(validate_callback_url_dependency),
    split_audio: bool = Query(
        default=False,
        description="Split stereo audio into separate channels for individual processing",
    ),
    repository: ITaskRepository = Depends(get_task_repository),
    file_service: FileService = Depends(get_file_service),
) -> Response:
    """
    Process an audio file from a URL for speech-to-text conversion.

    Args:
        background_tasks (BackgroundTasks): Background tasks dependency.
        model_params (WhisperModelParams): Whisper model parameters.
        align_params (AlignmentParams): Alignment parameters.
        diarize_params (DiarizationParams): Diarization parameters.
        asr_options_params (ASROptions): ASR options parameters.
        vad_options_params (VADOptions): VAD options parameters.
        url (str): URL of the audio file.
        callback_url (str | None): Optional URL to call back when processing is complete.
        split_audio (bool): If True, split stereo audio and process channels separately.
        repository (ITaskRepository): Task repository dependency.
        file_service (FileService): File service dependency.

    Returns:
        Response: Confirmation message of task queuing.
    """
    logger.info("Received URL for processing: %s", url)

    # Download file using file service
    temp_audio_file, filename = file_service.download_from_url(url)
    logger.info("File downloaded and saved temporarily: %s", temp_audio_file)

    # Validate extension
    file_service.validate_file_extension(temp_audio_file, ALLOWED_EXTENSIONS)

    return process_speech_to_text(
        temp_file=temp_audio_file,
        filename=filename,
        background_tasks=background_tasks,
        model_params=model_params,
        align_params=align_params,
        diarize_params=diarize_params,
        asr_options=asr_options_params,
        vad_options=vad_options_params,
        repository=repository,
        split_audio=split_audio,
        callback_url=callback_url,
        url=url,
    )
