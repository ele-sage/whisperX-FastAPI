"""This module provides functions for processing audio files."""

import subprocess
from tempfile import NamedTemporaryFile
from typing import Any, Union
from pathlib import Path
import numpy as np
from whisperx import load_audio
from whisperx.audio import SAMPLE_RATE

from app.core.logging import logger
from app.files import VIDEO_EXTENSIONS, check_file_extension


def convert_video_to_audio(file: str) -> str:
    """
    Convert a video file to an audio file.

    Args:
        file (str): The path to the video file.

    Returns:
        str: The path to the audio file.
    """
    temp_filename = NamedTemporaryFile(delete=False).name
    subprocess.call(
        [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists"
            "-i",
            file,
            "-vn",
            "-ac",
            "1",  # Mono audio
            "-ar",
            "16000",  # Sample rate of 16kHz
            "-f",
            "wav",  # Output format WAV
            temp_filename,
        ]
    )
    return temp_filename


def process_audio_file(audio_file: str) -> np.ndarray[Any, np.dtype[np.float32]]:
    """
    Check file if it is audio file, if it is video file, convert it to audio file.

    Args:
        audio_file (str): The path to the audio file.
    Returns:
        Audio: The processed audio.
    """
    file_extension = check_file_extension(audio_file)
    if file_extension in VIDEO_EXTENSIONS:
        audio_file = convert_video_to_audio(audio_file)
    return load_audio(audio_file)  # type: ignore[no-any-return]


def get_audio_duration(audio: np.ndarray[Any, np.dtype[np.float32]]) -> float:
    """
    Get the duration of the audio file.

    Args:
        audio_file (str): The path to the audio file.
    Returns:
        float: The duration of the audio file.
    """
    return len(audio) / SAMPLE_RATE  # type: ignore[no-any-return]


def get_audio_duration_from_file(file_path: str) -> float:
    """
    Get the duration of an audio/video file using ffprobe without loading it into memory.

    Args:
        file_path: Path to the media file.

    Returns:
        float: Duration in seconds.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        logger.error(f"Could not parse duration from ffprobe output: {result.stdout}")
        return get_audio_duration(load_audio(file_path))

def split_stereo_to_mono(audio_file_path: Union[str, Path]) -> tuple[str, str]:
    """
    Splits a stereo audio file into two separate mono WAV files (left and right)
    using the 'channelsplit' filter in FFmpeg.

    Args:
        audio_file_path (Union[str, Path]): Path to the stereo audio file.
    Returns:
        tuple[str, str]: Paths to the left and right mono audio files.
    """
    input_path = Path(audio_file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    left_channel_file = NamedTemporaryFile(delete=False, suffix=".wav").name
    right_channel_file = NamedTemporaryFile(delete=False, suffix=".wav").name

    command = [
        'ffmpeg',
        '-i', str(input_path),
        # Use the channelsplit filter. It takes one input (stereo) and creates two outputs ([left] and [right]).
        '-filter_complex', '[0:a]channelsplit=channel_layout=stereo[left][right]',
        '-map', '[left]',
        '-ac', '1',
        '-ar', '16000',
        '-y',
        left_channel_file,
        '-map', '[right]',
        '-ac', '1',
        '-ar', '16000',
        '-y',
        right_channel_file,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return left_channel_file, right_channel_file
    except FileNotFoundError:
        logger.error("`ffmpeg` command not found. Ensure FFmpeg is installed and in your system's PATH.")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg command failed with exit code {e.returncode}.")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        raise