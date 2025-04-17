"""
Optional helper functions for audio extraction, sentence splitting, etc.
"""

import ffmpeg
import nltk
from pathlib import Path
import subprocess # Added for better ffmpeg error handling
import datetime # Added for timestamp formatting

def format_timestamp(seconds: float) -> str:
    """Formats seconds into HH:MM:SS.fff format."""
    delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = delta.microseconds // 1000
    return f"[{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}]"

def extract_audio_ffmpeg(video_path, audio_path):
    """
    Extract audio from video file using ffmpeg and save as wav.
    Args:
        video_path (str or Path): Path to video file.
        audio_path (str or Path): Path to output wav file.
    Raises:
        RuntimeError: If ffmpeg fails.
    """
    try:
        process = (
            ffmpeg
            .input(str(video_path))
            .output(str(audio_path), format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run_async(pipe_stderr=True)
        )
        _, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {video_path}: {stderr.decode('utf-8')}")
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffmpeg error during setup for {video_path}: {e.stderr.decode('utf-8')}") from e

def split_text_into_sentences(text):
    """
    Split text into sentences using nltk's sent_tokenize.
    Args:
        text (str): The transcript text.
    Returns:
        List[str]: List of sentences.
    """
    # Ensure punkt is downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    return nltk.sent_tokenize(text)
