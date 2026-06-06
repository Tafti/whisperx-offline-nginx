#!/usr/bin/env python
"""Download required models for offline server deployment.

This script downloads:
1) Systran/faster-whisper-large-v3
2) jonatasgrosman/wav2vec2-large-xlsr-53-persian
"""

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm


DEFAULT_MODELS = [
    "Systran/faster-whisper-large-v3",
    "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download WhisperX server models")
    parser.add_argument(
        "--cache-dir",
        default="./data/models",
        help="Local cache directory for Hugging Face models",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries per model",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel download workers per model",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model repo id to download. Repeat this argument for multiple models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh from remote even when files are already cached",
    )
    return parser.parse_args()


def set_cache_env(cache_dir: Path) -> None:
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")


def download_model(repo_id: str, cache_dir: Path, retries: int, workers: int, force: bool) -> bool:
    for attempt in range(1, retries + 1):
        print("\n" + "=" * 72)
        print(f"Downloading {repo_id} (attempt {attempt}/{retries})")
        print("=" * 72)
        try:
            local_dir = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=workers,
                force_download=force,
                tqdm_class=tqdm,
            )
            print(f"OK: {repo_id}")
            print(f"Cached at: {local_dir}")
            return True
        except Exception as exc:
            print(f"ERROR: {repo_id} failed: {exc}")
            if attempt < retries:
                wait = 2 ** (attempt - 1)
                print(f"Retrying in {wait} seconds...")
                time.sleep(wait)
    return False


def main() -> int:
    args = parse_args()
    models = args.models if args.models else DEFAULT_MODELS
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_cache_env(cache_dir)

    print("\nWhisperX offline model download")
    print(f"Cache directory: {cache_dir}")
    print("Models:")
    for model in models:
        print(f"- {model}")

    failures = []
    for model in models:
        if not download_model(
            repo_id=model,
            cache_dir=cache_dir,
            retries=args.retries,
            workers=args.workers,
            force=args.force,
        ):
            failures.append(model)

    if failures:
        print("\nDownload completed with errors.")
        print("Failed models:")
        for model in failures:
            print(f"- {model}")
        return 1

    total_size = sum(path.stat().st_size for path in cache_dir.rglob("*") if path.is_file())
    size_gb = total_size / (1024 ** 3)
    print("\nAll required models downloaded successfully.")
    print(f"Total cache size: {size_gb:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())