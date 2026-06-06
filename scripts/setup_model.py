#!/usr/bin/env python
"""Compatibility wrapper for model setup.

This script delegates to scripts/download_model.py and downloads all models
required by this project for offline server deployment.
"""

import subprocess
import sys
from pathlib import Path


def setup_model() -> bool:
    project_root = Path(__file__).resolve().parent.parent
    downloader = project_root / "scripts" / "download_model.py"

    cmd = [
        sys.executable,
        str(downloader),
        "--cache-dir",
        str(project_root / "data" / "models"),
    ]

    print("Running model setup for offline server deployment...")
    result = subprocess.run(cmd, cwd=str(project_root))
    return result.returncode == 0


if __name__ == "__main__":
    raise SystemExit(0 if setup_model() else 1)