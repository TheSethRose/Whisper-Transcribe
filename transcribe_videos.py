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
import time

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

def process_video(video_file, config, worker_id, status_dict):
    try:
        status_dict[worker_id] = f"{video_file.name}: 0% (Extracting audio)"
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / (video_file.stem + '.wav')
            extract_audio_ffmpeg(video_file, audio_path)

            status_dict[worker_id] = f"{video_file.name}: 0% (Transcribing)"
            cli_path = config['cli_path']
            models_dir = config['models_dir']
            results_dir = config['results_dir']
            add_timestamps = config['timestamps']
            language = config['language']
            cmd = [
                cli_path,
                "transcribe",
                "--audio-path", str(audio_path),
                "--model-path", str(models_dir),
                "--text-decoder-compute-units", "cpuAndNeuralEngine",
                "--audio-encoder-compute-units", "cpuAndNeuralEngine",
                "--report-path", str(results_dir), "--report",
            ]
            if language:
                cmd.extend(["--language", language])
            # Use Popen to capture real-time output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
            try:
                if process.stdout is None:
                    process.kill()
                    status_dict[worker_id] = f"Error: {video_file.name}"
                    raise RuntimeError(f"whisperkit-cli failed to start for {video_file.name}: stdout is None")
                for line in process.stdout:
                    line = line.strip()
                    if line.startswith("PROGRESS: "):
                        try:
                            percent = int(line.split(": ")[1])
                            status_dict[worker_id] = f"{video_file.name}: {percent}% (Transcribing)"
                        except Exception:
                            pass
                process.wait(timeout=600)
            except Exception as e:
                process.kill()
                status_dict[worker_id] = f"Error: {video_file.name}"
                raise RuntimeError(f"whisperkit-cli failed for {video_file.name}: {e}")
            if process.returncode != 0:
                status_dict[worker_id] = f"Error: {video_file.name}"
                raise RuntimeError(f"whisperkit-cli failed for {video_file.name}")
            result_json_path = results_dir / (video_file.stem + '.json')
            if not result_json_path.exists():
                status_dict[worker_id] = f"Error: {video_file.name}"
                raise RuntimeError(f"whisperkit-cli did not produce expected JSON report at {result_json_path}")

            status_dict[worker_id] = f"{video_file.name}: 66% (Formatting output)"
            with open(result_json_path, "r", encoding='utf-8') as f:
                results = json.load(f)
            if 'segments' not in results or not results['segments']:
                transcript_text = results.get('text', '')
                sentences = split_text_into_sentences(transcript_text)
                formatted_lines = [s.strip() for s in sentences if s.strip()]
            else:
                formatted_lines = []
                for segment in results['segments']:
                    text = segment.get('text', '').strip()
                    text = text.replace('<|startoftranscript|>', '').replace('<|endoftext|>', '')
                    text = text.replace('<|en|>', '').replace('<|transcribe|>', '').replace('<|nocaptions|>', '')
                    text = subprocess.run(['sed', r's/<|\([0-9]*\.[0-9]*\)|>//g'], input=text, capture_output=True, text=True).stdout.strip()
                    if not text:
                        continue
                    sentences = split_text_into_sentences(text)
                    start_time_str = format_timestamp(segment.get('start', 0.0)) if add_timestamps else ""
                    for sentence in sentences:
                        clean_sentence = sentence.strip()
                        if clean_sentence:
                            formatted_lines.append(f"{start_time_str} {clean_sentence}".strip())
            out_txt = config['output_folder'] / (video_file.stem + '.txt')
            with open(out_txt, 'w', encoding='utf-8') as f:
                for line in formatted_lines:
                    f.write(line + '\n')
        status_dict[worker_id] = f"{video_file.name}: 100% (Done)"
        return None
    except Exception as e:
        status_dict[worker_id] = f"Error: {video_file.name}"
        return (video_file.name, e)

def main():
    args = parse_args()
    config = load_config(args)
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
    try:
        from whisperkit.pipelines import WhisperKit
        pipe_init = WhisperKit(config['model_path'], out_dir=str(config['cache_dir']))
        config['cli_path'] = pipe_init.cli_path
        config['models_dir'] = pipe_init.models_dir
        config['results_dir'] = Path(pipe_init.results_dir)
    except Exception as e:
        print(f"[FATAL] Error initializing WhisperKit: {e}")
        print("Please ensure WHISPERKIT_MODEL_PATH is correct and dependencies are installed.")
        sys.exit(1)
    all_video_files = find_video_files(config['input_folder'])
    if not all_video_files:
        print("[INFO] No supported video files found in input folder.")
        return
    video_files_to_process = []
    if config['overwrite']:
        video_files_to_process = all_video_files
        print(f"[INFO] Found {len(all_video_files)} video(s). Overwriting enabled.")
    else:
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
    num_workers = config['num_workers']
    print(f"[INFO] Starting transcription with {num_workers} worker(s)...\n")
    manager = multiprocessing.Manager()
    status_dict = manager.dict({i: "Idle" for i in range(num_workers)})
    errors = manager.list()
    pool = multiprocessing.Pool(processes=num_workers)
    results = []
    for idx, video_file in enumerate(video_files_to_process):
        worker_id = idx % num_workers
        result = pool.apply_async(process_video, args=(video_file, config, worker_id, status_dict), callback=lambda r: errors.append(r) if r else None)
        results.append(result)
    completed = 0
    total = len(video_files_to_process)
    while completed < total:
        os.system('clear' if os.name == 'posix' else 'cls')
        print("Worker Status Table:")
        print("====================")
        for i in range(num_workers):
            print(f"Worker {i+1}: {status_dict[i]}")
        completed = sum(1 for r in results if r.ready())
        print(f"\nProgress: {completed}/{total} videos processed.")
        time.sleep(0.01)
    pool.close()
    pool.join()
    # --- Report Errors ---
    real_errors = [e for e in errors if e]
    if real_errors:
        print("\n--- Errors Occurred ---")
        for filename, error in real_errors:
            print(f"[ERROR] {filename}: {error}")
        print("-----------------------")
    print("\n[INFO] All done.\n")

if __name__ == "__main__":
    # Set start method for multiprocessing (recommended for macOS/Windows)
    multiprocessing.set_start_method("spawn", force=True)
    main()
