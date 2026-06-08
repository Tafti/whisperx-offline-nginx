#!/usr/bin/env python
"""
Download WhisperX models with progress bars and retry logic.
Run this script before starting the FastAPI server.
"""

import os
import time
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from tqdm import tqdm

# ---------- Configuration ----------
MODEL_CACHE_ROOT = Path("./data/models")
MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# Set cache directory
os.environ["HF_HOME"] = str(MODEL_CACHE_ROOT)
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_ROOT / "hub")

# Custom progress callback
class ProgressCallback:
    """Display download progress with tqdm"""
    def __init__(self, description="Downloading"):
        self.description = description
        self.progress_bar = None
        self.total_size = 0
        self.downloaded = 0
    
    def __call__(self, current, total):
        if self.progress_bar is None:
            self.total_size = total
            self.progress_bar = tqdm(
                total=total,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=self.description,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]'
            )
        
        # Update progress
        delta = current - self.downloaded
        self.progress_bar.update(delta)
        self.downloaded = current
        
        if current >= total:
            self.progress_bar.close()
            print(f"\n✅ {self.description} complete!")

def download_with_progress(model_name, max_retries=3):
    """Download a model with progress bars and retry logic"""
    
    for attempt in range(max_retries):
        print(f"\n{'='*60}")
        print(f"📥 Attempt {attempt+1}/{max_retries}: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Create progress callback
            callback = ProgressCallback(f"Downloading {model_name.split('/')[-1]}")
            
            # Download with progress tracking
            local_dir = snapshot_download(
                repo_id=model_name,
                cache_dir=str(MODEL_CACHE_ROOT),
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=4,
                tqdm_class=tqdm,  # Enable built-in tqdm
                # Custom progress not directly supported, but tqdm_class works
            )
            
            print(f"\n✅ Successfully downloaded {model_name} to:")
            print(f"   {local_dir}")
            return True
            
        except Exception as e:
            print(f"\n❌ Attempt {attempt+1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                
                # Show countdown
                for i in range(wait_time, 0, -1):
                    print(f"   Retrying in {i} seconds...", end='\r')
                    time.sleep(1)
                print(" " * 30, end='\r')  # Clear the line
            else:
                print(f"\n❌ Failed to download {model_name} after {max_retries} attempts")
    
    return False

def download_specific_file_with_progress():
    """Alternative: Download a specific file with detailed progress"""
    model_name = "guillaumekln/faster-whisper-large-v2"
    filename = "model.bin"
    
    print(f"\n📥 Downloading {filename} from {model_name}...")
    
    try:
        # This shows a simple progress bar
        file_path = hf_hub_download(
            repo_id=model_name,
            filename=filename,
            cache_dir=str(MODEL_CACHE_ROOT),
            resume=True,
            local_files_only=False
        )
        print(f"✅ Downloaded to: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def check_existing_model(model_name):
    """Check if model already exists locally"""
    model_path = MODEL_CACHE_ROOT / "hub" / model_name.replace("/", "--")
    
    if model_path.exists():
        # Count files
        files = list(model_path.rglob("*"))
        if files:
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            size_mb = total_size / (1024 * 1024)
            print(f"\n📦 Found existing model at: {model_path}")
            print(f"   Size: {size_mb:.2f} MB")
            print(f"   Files: {len([f for f in files if f.is_file()])}")
            return True
    
    return False

def main():
    """Main download function with user selection"""
    print("\n" + "="*60)
    print("🎙️  WhisperX Model Downloader with Progress Bars")
    print("="*60)
    
    models = {
        "1": ("guillaumekln/faster-whisper-large-v2", "Large v2 (2.9 GB) - Best quality, slower"),
        "2": ("Systran/faster-whisper-large-v3", "Large v3 (2.9 GB) - Latest version"),
        "3": ("guillaumekln/faster-whisper-medium", "Medium (1.5 GB) - Balanced"),
        "4": ("guillaumekln/faster-whisper-small", "Small (0.5 GB) - Fast, reasonable quality"),
        "5": ("guillaumekln/faster-whisper-base", "Base (0.15 GB) - Fastest, lower quality"),
    }
    
    print("\nAvailable models:")
    for key, (name, desc) in models.items():
        # Check if already downloaded
        exists = check_existing_model(name)
        status = "✅" if exists else "⬜"
        print(f"  {key}. {status} {desc}")
        print(f"       Repo: {name}")
    
    choice = input("\n📌 Select model (1-5) [default: 1]: ").strip() or "1"
    
    if choice not in models:
        print("❌ Invalid choice. Using default (Large v2)")
        choice = "1"
    
    model_name, description = models[choice]
    print(f"\n📦 Selected: {description}")
    
    # Check if already exists
    if check_existing_model(model_name):
        overwrite = input("\n⚠️  Model already exists. Download again? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("✅ Using existing model. No download needed.")
            return
    
    # Download the model
    print(f"\n🚀 Starting download of {model_name}...")
    print("   (This may take several minutes depending on your connection speed)\n")
    
    start_time = time.time()
    success = download_with_progress(model_name)
    elapsed = time.time() - start_time
    
    if success:
        print(f"\n{'='*60}")
        print(f"✅ Download completed in {elapsed:.2f} seconds!")
        print(f"📁 Model stored in: {MODEL_CACHE_ROOT}")
        print(f"{'='*60}")
        
        # Show cache size
        total_size = sum(f.stat().st_size for f in MODEL_CACHE_ROOT.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        print(f"\n💾 Total cache size: {size_gb:.2f} GB")
        
    else:
        print(f"\n{'='*60}")
        print("❌ Download failed after multiple attempts")
        print("="*60)
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Try a mirror: export HF_ENDPOINT=https://hf-mirror.com")
        print("3. Use a smaller model (options 3-5)")
        print("4. Download manually from: https://huggingface.co/" + model_name)
        sys.exit(1)

def download_with_tqdm_workaround():
    """
    Alternative method using subprocess with tqdm
    This works better for very large files
    """
    import subprocess
    
    model_name = "guillaumekln/faster-whisper-large-v2"
    print(f"\n📥 Downloading {model_name} with external tool...")
    
    # Use huggingface-cli with tqdm
    cmd = [
        sys.executable, "-m", "huggingface_hub.commands.huggingface_cli",
        "download", model_name,
        "--resume-download",
        "--quiet"  # Remove this to see more output
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Simple progress indicator
    with tqdm(total=100, desc="Downloading", unit='%') as pbar:
        for line in process.stdout:
            if "Downloading" in line:
                pbar.update(1)
            print(line, end='')
    
    process.wait()
    return process.returncode == 0

if __name__ == "__main__":
    try:
        # Try to import tqdm
        from tqdm import tqdm
        main()
    except ImportError:
        print("⚠️  tqdm not installed. Installing...")
        os.system(f"{sys.executable} -m pip install tqdm")
        from tqdm import tqdm
        main()