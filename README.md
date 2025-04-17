# Batch Video Transcription Tool

This tool batch transcribes all video files in a specified folder using a local WhisperKit CoreML model (v2 or v3) on Apple Silicon (macOS). Transcripts are grouped by sentences and saved as .txt files in a configurable output folder (default: `transcriptions/` subfolder of your input folder).

## Features
- Batch process all videos in a folder
- Uses local WhisperKit CoreML models (v2/v3, user-selectable)
- Fast, on-device transcription (Apple Silicon optimized)
- Output grouped by sentences (one per line)
- Optional timestamps per sentence/segment
- Output `.txt` files in a configurable output folder
- Parallel processing for speed (configurable number of workers)
- Skips existing transcripts by default (configurable)
- Progress bar and error reporting
- All configuration via `.env` file or CLI overrides

## Requirements
- macOS (Apple Silicon, e.g., M1/M2/M3)
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (for dependency management and running scripts)
- [ffmpeg](https://ffmpeg.org/) (must be installed and in your PATH)
- WhisperKit-compatible CoreML model (see below)
- [python-dotenv](https://github.com/theskumar/python-dotenv) (for .env support, loaded automatically)
- [tqdm](https://github.com/tqdm/tqdm) (for progress bar)

## Setup
1. **Install uv** (if not already):
   ```sh
   pip install uv
   # or follow instructions at https://github.com/astral-sh/uv
   ```
2. **Install ffmpeg** (if not already):
   ```sh
   brew install ffmpeg
   # or download from https://ffmpeg.org/download.html
   ```
3. **Clone this repository and install dependencies:**
   ```sh
   uv pip install -r requirements.txt
   ```
4. **Download or place your WhisperKit CoreML model** (e.g., v2 or v3) somewhere on your system. Example paths:
   - `/Users/youruser/Library/Application Support/MacWhisper/models/whisperkit/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v2`
   - `/Users/youruser/Library/Application Support/MacWhisper/models/whisperkit/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3-v20240930_turbo`

## Usage
1. **Configure your `.env` file (recommended):**
   - Copy `.env_example` to `.env` and edit as needed:
     ```sh
     cp .env_example .env
     # Then edit .env in your favorite editor
     ```
   - Example `.env`:
     ```env
     WHISPERKIT_MODEL_PATH=openai/whisper-large-v2
     INPUT_FOLDER=/absolute/path/to/your/video/folder
     OUTPUT_FOLDER=
     NUM_WORKERS=4
     OVERWRITE=false
     TIMESTAMPS=false
     LANGUAGE=
     ```
   - If `OUTPUT_FOLDER` is empty, transcripts will be saved in `INPUT_FOLDER/transcriptions/`.
   - If `NUM_WORKERS` is not set, defaults to half your CPU cores.
   - If `OVERWRITE` is false, existing transcripts are skipped.
   - If `TIMESTAMPS` is true, each line in the transcript will be prepended with a timestamp.
   - If `LANGUAGE` is set (e.g., `en`), it will be passed to the model for improved accuracy.

2. **Run the batch transcription script:**
   - **With uv:**
     ```sh
     uv run transcribe_videos.py
     # or, with CLI overrides:
     uv run transcribe_videos.py /path/to/your/video/folder --num-workers 2 --overwrite --timestamps
     ```

## Environment Variables
| Variable Name           | Description                                                                                       | Example Value |
|------------------------|---------------------------------------------------------------------------------------------------|---------------|
| `WHISPERKIT_MODEL_PATH`| Hugging Face identifier for the WhisperKit CoreML model to use for transcription.                 | `openai/whisper-large-v2` |
| `INPUT_FOLDER`         | Absolute path to the folder containing video files.                                               | `/Users/youruser/Videos` |
| `OUTPUT_FOLDER`        | Absolute path to output folder for transcripts. If empty, uses `INPUT_FOLDER/transcriptions/`.   | `/Users/youruser/Transcripts` or empty |
| `NUM_WORKERS`          | Number of parallel processes to use. If not set, defaults to half your CPU cores.                | `4`           |
| `OVERWRITE`            | Whether to overwrite existing transcript files (`true`/`false`).                                 | `false`       |
| `TIMESTAMPS`           | Whether to prepend timestamps to each transcript line (`true`/`false`).                         | `false`       |
| `LANGUAGE`             | Language code for transcription (e.g., `en`, `fr`). Optional.                                   | `en`          |

## File Structure
```
project-root/
├── transcribe_videos.py         # Main Python script (CLI entry point, now with shebang for direct execution)
├── requirements.txt             # Python dependencies (whisperkittools, ffmpeg-python, nltk, python-dotenv, tqdm, etc.)
├── README.md                    # Setup, usage, and environment variable instructions
├── utils.py                     # (Optional) Helper functions for audio extraction, sentence splitting, etc.
├── .env                         # (Optional) Environment variable file for model path and settings
├── .env_example                 # Example .env file
├── .gitignore                   # Git ignore file
└── <input_folder>/              # User-specified folder containing video files (not part of repo)
    └── transcriptions/          # Output folder for .txt transcripts (created at runtime)
```

## License
MIT
