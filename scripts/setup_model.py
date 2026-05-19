#!/usr/bin/env python
"""
Manual model setup script - downloads model and places it correctly
Run this BEFORE starting the server
"""
import os
import sys
from pathlib import Path
import subprocess

def setup_model():
    print("🔧 Setting up WhisperX model...")
    
    # Set cache directory
    cache_dir = Path("./data/models")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    
    model_name = "base"  # or "small", "medium", "large-v2"
    
    print(f"📥 Downloading {model_name} model...")
    
    # Use Python to download with offline fallback
    cmd = [
        sys.executable, "-c", f"""
import os
os.environ['HF_HOME'] = '{cache_dir}'
from huggingface_hub import snapshot_download
try:
    # Try the official faster-whisper models
    snapshot_download('guillaumekln/faster-whisper-{model_name}', 
                     cache_dir='{cache_dir}',
                     local_dir_use_symlinks=False,
                     resume_download=True)
    print('✅ Download complete')
except Exception as e:
    print(f'Error: {{e}}')
    sys.exit(1)
"""
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n✅ Model setup complete!")
        print(f"📁 Model location: {cache_dir}")
        
        # Verify download
        model_path = cache_dir / "hub" / f"models--guillaumekln--faster-whisper-{model_name}"
        if model_path.exists():
            size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            size_mb = size / (1024 * 1024)
            print(f"📊 Model size: {size_mb:.2f} MB")
        return True
    else:
        print(f"\n❌ Setup failed")
        return False

if __name__ == "__main__":
    setup_model()