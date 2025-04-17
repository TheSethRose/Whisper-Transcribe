#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import shutil
import tempfile
import multiprocessing
import json
import subprocess
import datetime
from functools import partial
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env FIRST
load_dotenv()

# Assuming utils.py exists with updated functions:
# - extract_audio_ffmpeg(video_path, audio_path) -> raises RuntimeError on failure
# - split_text_into_sentences(text) -> List[str]
# - format_timestamp(seconds) -> str like '[HH:MM:SS.fff]'
from utils import extract_audio_ffmpeg, split_text_into_sentences, format_timestamp


SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv'}
DEFAULT_MODEL_PATH = "openai/whisper-large-v2" # Default if not in .env

def str_to_bool(value):
    """Converts string representations of truth to bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return value.lower() in ('true', '1', 't', 'y', 'yes')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch transcribe videos using WhisperKit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_folder", type=str, nargs='?', default=None,
        help="Path to folder containing video files (overrides INPUT_FOLDER in .env)."
    )
    parser.add_argument(
        "--output-folder", type=str, default=None,
        help="Path to folder for saving transcripts (overrides OUTPUT_FOLDER in .env; defaults to INPUT_FOLDER/transcriptions)."
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of parallel processes (overrides NUM_WORKERS in .env; defaults to half CPU cores)."
    )
    parser.add_argument(
        "--overwrite", action='store_true', default=None,
        help="Overwrite existing transcript files (overrides OVERWRITE=true in .env)."
    )
    parser.add_argument(
        "--skip-overwrite", action='store_false', dest='overwrite',
        help="Explicitly skip existing files (overrides OVERWRITE=false in .env)."
    )
    parser.add_argument(
        "--timestamps", action='store_true', default=None,
        help="Include timestamps in output files (overrides TIMESTAMPS=true in .env)."
    )
    parser.add_argument(
        "--no-timestamps", action='store_false', dest='timestamps',
        help="Do not include timestamps (overrides TIMESTAMPS=false in .env)."
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Language code (e.g., 'en', 'es') for transcription (overrides LANGUAGE in .env)."
    )
    return parser.parse_args()

def load_config(args):
    config = {}

    # --- Load from environment (with defaults) ---
    config['model_path'] = os.environ.get("WHISPERKIT_MODEL_PATH", DEFAULT_MODEL_PATH)
    config['input_folder'] = os.environ.get("INPUT_FOLDER")
    config['output_folder'] = os.environ.get("OUTPUT_FOLDER")

    cpu_cores = os.cpu_count()
    default_workers = max(1, cpu_cores // 2) if cpu_cores else 1 # Handle None case
    try:
        env_workers = os.environ.get("NUM_WORKERS")
        config['num_workers'] = int(env_workers) if env_workers else default_workers
    except ValueError:
        print(f"Warning: Invalid NUM_WORKERS '{env_workers}' in .env, using default: {default_workers}")
        config['num_workers'] = default_workers

    config['overwrite'] = str_to_bool(os.environ.get("OVERWRITE", False))
    config['timestamps'] = str_to_bool(os.environ.get("TIMESTAMPS", False))
    config['language'] = os.environ.get("LANGUAGE") or None # Ensure None if empty string

    # --- Override with CLI args ---
    if args.input_folder:
        config['input_folder'] = args.input_folder
    if args.output_folder:
        config['output_folder'] = args.output_folder
    if args.num_workers is not None:
        config['num_workers'] = args.num_workers
    if args.overwrite is not None:
        config['overwrite'] = args.overwrite
    if args.timestamps is not None:
        config['timestamps'] = args.timestamps
    if args.language:
        config['language'] = args.language

    # --- Validate and Finalize ---
    if not config['input_folder']:
        print("Error: Input folder must be specified via CLI argument or INPUT_FOLDER in .env file.")
        sys.exit(1)

    config['input_folder'] = Path(config['input_folder']).resolve()
    if not config['input_folder'].is_dir():
        print(f"Error: Input folder not found or is not a directory: {config['input_folder']}")
        sys.exit(1)

    if not config['output_folder']:
        config['output_folder'] = config['input_folder'] / 'transcriptions'
    else:
        config['output_folder'] = Path(config['output_folder']).resolve()

    if not config['model_path']:
         raise EnvironmentError("WHISPERKIT_MODEL_PATH must be set via CLI or .env.")

    config['cache_dir'] = Path('.whisperkit_cache').resolve()

    return config

def find_video_files(folder):
    return [f for f in Path(folder).iterdir() if f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS and f.is_file()]

def process_video(video_file, config):
    """Worker function to process a single video file."""
    output_folder = config['output_folder']
    cli_path = config['cli_path']
    models_dir = config['models_dir'] # Path determined by WhisperKit init
    results_dir = config['results_dir'] # Path determined by WhisperKit init
    add_timestamps = config['timestamps']
    language = config['language']

    out_txt = output_folder / (video_file.stem + '.txt')
    result_json_path = results_dir / (video_file.stem + '.json')

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / (video_file.stem + '.wav')

            # 1. Extract Audio
            extract_audio_ffmpeg(video_file, audio_path)

            # 2. Transcribe using WhisperKit CLI
            cmd = [
                cli_path,
                "transcribe",
                "--audio-path", str(audio_path),
                "--model-path", str(models_dir),
                # Use defaults for compute units for now
                "--text-decoder-compute-units", "cpuAndNeuralEngine",
                "--audio-encoder-compute-units", "cpuAndNeuralEngine",
                "--report-path", str(results_dir), "--report",
            ]
            if language:
                # Assuming WhisperKit CLI uses --language
                # Check `swift run whisperkit-cli transcribe --help` if needed
                cmd.extend(["--language", language])

            # If we want word timestamps in JSON, add flag here (but we format from segment times)
            # cmd.append("--word-timestamps")

            process = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if process.returncode != 0:
                print(f"[ERROR] whisperkit-cli failed for {video_file.name}:\nSTDERR: {process.stderr.strip()}\nSTDOUT: {process.stdout.strip()}")
                raise RuntimeError(f"whisperkit-cli failed for {video_file.name}")

            # 3. Read JSON result
            if not result_json_path.exists():
                 raise RuntimeError(f"whisperkit-cli did not produce expected JSON report at {result_json_path}")

            with open(result_json_path, "r", encoding='utf-8') as f:
                results = json.load(f)

            if 'segments' not in results or not results['segments']:
                print(f"[WARNING] No segments found in transcription for {video_file.name}")
                transcript_text = results.get('text', '') # Use full text if no segments
                sentences = split_text_into_sentences(transcript_text)
                formatted_lines = [s.strip() for s in sentences if s.strip()]
            else:
                 # 4. Format Output
                formatted_lines = []
                for segment in results['segments']:
                    text = segment.get('text', '').strip()
                    # Remove special tokens if present (may vary by model/version)
                    text = text.replace('<|startoftranscript|>', '').replace('<|endoftext|>', '')
                    text = text.replace('<|en|>', '').replace('<|transcribe|>', '').replace('<|nocaptions|>', '')
                    # Remove timestamp tokens like <|0.00|>
                    text = subprocess.run(['sed', r's/<|\([0-9]*\.[0-9]*\)|>//g'], input=text, capture_output=True, text=True).stdout.strip()

                    if not text:
                        continue

                    sentences = split_text_into_sentences(text)
                    start_time_str = format_timestamp(segment.get('start', 0.0)) if add_timestamps else ""

                    for sentence in sentences:
                        clean_sentence = sentence.strip()
                        if clean_sentence:
                             formatted_lines.append(f"{start_time_str} {clean_sentence}".strip())

            # 5. Save Transcript
            with open(out_txt, 'w', encoding='utf-8') as f:
                for line in formatted_lines:
                    f.write(line + '\n')

        return None # Success
    except Exception as e:
        return (video_file.name, e) # Failure

def main():
    args = parse_args()
    config = load_config(args)

    # --- Initial Setup (Run once in main process) ---
    config['output_folder'].mkdir(parents=True, exist_ok=True)
    config['cache_dir'].mkdir(parents=True, exist_ok=True)

    print("\n========== WhisperKit Batch Video Transcription ==========")
    print(f"Input folder:      {config['input_folder']}")
    print(f"Output folder:     {config['output_folder']}")
    print(f"Model:             {config['model_path']}")
    print(f"Workers:           {config['num_workers']}")
    print(f"Overwrite:         {config['overwrite']}")
    print(f"Timestamps:        {config['timestamps']}")
    print(f"Language:          {config['language'] if config['language'] else 'auto-detect'}")
    print("========================================================\n")

    # Import WhisperKit only when needed
    try:
        from whisperkit.pipelines import WhisperKit
        # Initialize WhisperKit primarily to ensure CLI is built and get paths
        # This will clone/build/download models if not already cached
        pipe_init = WhisperKit(config['model_path'], out_dir=str(config['cache_dir']))
        config['cli_path'] = pipe_init.cli_path
        config['models_dir'] = pipe_init.models_dir
        config['results_dir'] = pipe_init.results_dir # Where the CLI saves JSON reports
        print("WhisperKit initialized.")
    except Exception as e:
        print(f"[FATAL] Error initializing WhisperKit: {e}")
        print("Please ensure WHISPERKIT_MODEL_PATH is correct and dependencies are installed.")
        sys.exit(1)

    # --- Find and Filter Videos ---
    all_video_files = find_video_files(config['input_folder'])
    if not all_video_files:
        print("[INFO] No supported video files found in input folder.")
        return # Changed from sys.exit(1)

    video_files_to_process = []
    if config['overwrite']:
        video_files_to_process = all_video_files
        print(f"[INFO] Found {len(all_video_files)} video(s). Overwriting enabled.")
    else:
        print("[INFO] Checking for existing transcripts (overwrite disabled)...")
        for vf in all_video_files:
            out_txt = config['output_folder'] / (vf.stem + '.txt')
            if not out_txt.exists():
                video_files_to_process.append(vf)
            else:
                 print(f"  [SKIP] {vf.name} (transcript exists at {out_txt})")
        print(f"[INFO] Found {len(all_video_files)} total video(s). Processing {len(video_files_to_process)}.")

    if not video_files_to_process:
        print("[INFO] No videos to process. Exiting.")
        return

    # --- Parallel Processing ---
    num_workers = config['num_workers']
    print(f"[INFO] Starting transcription with {num_workers} worker(s)...\n")

    # Use functools.partial to pass the config to the worker
    worker_func = partial(process_video, config=config)

    errors = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap_unordered for progress updates as tasks complete
        results_iterator = pool.imap_unordered(worker_func, video_files_to_process)
        for result in tqdm(results_iterator, total=len(video_files_to_process), desc="Transcribing", ncols=80):
            if result is not None: # An error occurred
                errors.append(result)

    # --- Report Errors ---
    if errors:
        print("\n--- Errors Occurred ---")
        for filename, error in errors:
            print(f"[ERROR] {filename}: {error}")
        print("-----------------------")

    print("\n[INFO] All done.\n")

if __name__ == "__main__":
    # Set start method for multiprocessing (recommended for macOS/Windows)
    multiprocessing.set_start_method("spawn", force=True)
    main()
