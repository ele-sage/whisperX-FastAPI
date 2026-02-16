"""Callback functionality."""

import time
from datetime import datetime
from typing import Any

import httpx
from fastapi import HTTPException, status
from pydantic import HttpUrl

from app.core.config import get_settings
from app.core.logging import logger
from app.schemas import Metadata, Result


def validate_callback_url(callback_url: str) -> bool:
    """
    Validate that a callback URL is reachable.

    Args:
        callback_url: The callback URL to validate

    Returns:
        bool: True if the URL is reachable, False otherwise
    """
    try:
        with httpx.Client(
            timeout=float(get_settings().callback.CALLBACK_TIMEOUT)
        ) as client:
            response = client.head(callback_url)
            if response.status_code < 400 or response.status_code == 405:
                return True
            else:
                logger.warning(
                    "Callback URL returned server error %d: %s",
                    response.status_code,
                    callback_url,
                )
                return False
    except httpx.ConnectError:
        logger.warning(
            "Callback URL not reachable (connection error): %s", callback_url
        )
        return False
    except httpx.TimeoutException:
        logger.warning("Callback URL validation timeout: %s", callback_url)
        return False
    except Exception as e:
        logger.warning("Callback URL validation failed: %s - %s", callback_url, str(e))
        return False


def validate_callback_url_dependency(
    callback_url: HttpUrl | None = None,
) -> str | None:
    """
    Fastapi dependency to validate callback URL during request validation.

    Args:
        callback_url: Optional callback URL from request

    Returns:
        str | None: Validated callback URL as string, or None if not provided

    Raises:
        HTTPException: If callback URL validation fails
    """
    if callback_url is None:
        return None

    callback_url_str = str(callback_url)

    if not validate_callback_url(callback_url_str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Callback URL is not reachable: {callback_url_str}",
        )

    return callback_url_str


def _serialize_datetime(obj: Any) -> Any:
    """Recursively serialize datetime objects to ISO format strings."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: _serialize_datetime(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_datetime(item) for item in obj]
    else:
        return obj


def post_task_callback(callback_url: str, payload: dict[str, Any]) -> None:
    """
    POST task callback with serialized datetime objects and retry logic.

    Args:
        callback_url: The URL to send the callback to
        payload: The payload to send
    """
    max_retries = get_settings().callback.CALLBACK_MAX_RETRIES

    serialized_payload = _serialize_datetime(payload)

    for attempt in range(max_retries):
        try:
            with httpx.Client(
                timeout=float(get_settings().callback.CALLBACK_TIMEOUT)
            ) as client:
                response = client.post(callback_url, json=serialized_payload)
                response.raise_for_status()

            logger.info(
                "Successfully posted callback to %s",
                callback_url,
            )
            return

        except httpx.TimeoutException as e:
            logger.warning(
                "Timeout posting callback to %s (attempt %d/%d): %s",
                callback_url,
                attempt + 1,
                max_retries,
                str(e),
            )
        except httpx.HTTPStatusError as e:
            logger.warning(
                "HTTP error posting callback to %s (attempt %d/%d): %s - %s",
                callback_url,
                attempt + 1,
                max_retries,
                e.response.status_code,
                e.response.text[:200],
            )
        except Exception as e:
            logger.warning(
                "Failed to POST callback to %s (attempt %d/%d): %s",
                callback_url,
                attempt + 1,
                max_retries,
                str(e),
            )

        if attempt < max_retries - 1:
            wait_time = 2**attempt
            logger.debug("Waiting %ss before retry...", wait_time)
            time.sleep(wait_time)

    logger.error(
        "All callback attempts failed for %s, after %d attempts",
        callback_url,
        max_retries,
    )


def send_task_result_callback(task: Any) -> None:
    """
    Construct the result payload and send the callback for a completed/failed task.

    Args:
        task: The task object (DomainTask) containing status, result, error, etc.
    """
    if not task.callback_url:
        return

    try:
        metadata = Metadata(
            identifier=task.uuid,
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
        post_task_callback(task.callback_url, result_payload.model_dump())
        
    except Exception as e:
        logger.error(
            "Failed to prepare/send callback for task %s: %s",
            task.uuid,
            str(e),
        )
