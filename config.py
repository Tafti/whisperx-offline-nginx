import os
import logging
import platform
from typing import Any, Dict
from pathlib import Path


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

# ---------- Paths ----------
BASE_DIR = Path(__file__).parent
load_env_file(BASE_DIR / ".env")
MODEL_CACHE_ROOT = BASE_DIR / "data" / "models"
MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# Set environment variables for Hugging Face and WhisperX caching
os.environ["HF_HOME"] = str(MODEL_CACHE_ROOT)
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_ROOT / "hub")
os.environ["WHISPERX_CACHE"] = str(MODEL_CACHE_ROOT / "whisperx")

# CRITICAL: Force offline mode - prevents any network requests
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Disable telemetry and reduce timeouts
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("whisperx_api")

# ---------- Device ----------
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"


def _bytes_to_gb(value: int) -> float:
    return round(value / (1024 ** 3), 2)


def _get_total_ram_bytes() -> int | None:
    # Windows memory detection without extra dependencies.
    if os.name == "nt":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        memory_status = MEMORYSTATUSEX()
        memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
            return int(memory_status.ullTotalPhys)
        return None

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * total_pages)
    except (AttributeError, ValueError, OSError):
        return None


def collect_hardware_info() -> Dict[str, Any]:
    cpu_count = os.cpu_count() or 0
    total_ram_bytes = _get_total_ram_bytes()

    info: Dict[str, Any] = {
        "device": device,
        "compute_type": compute_type,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python_version": platform.python_version(),
        "cpu_cores_logical": cpu_count,
        "ram_total_gb": _bytes_to_gb(total_ram_bytes) if total_ram_bytes else None,
    }

    if device == "cuda":
        gpu_index = torch.cuda.current_device()
        gpu_props = torch.cuda.get_device_properties(gpu_index)
        info.update({
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_index": gpu_index,
            "gpu_name": gpu_props.name,
            "gpu_memory_total_gb": _bytes_to_gb(gpu_props.total_memory),
            "gpu_capability": f"{gpu_props.major}.{gpu_props.minor}",
            "gpu_multi_processor_count": gpu_props.multi_processor_count,
        })
    else:
        info.update({
            "cuda_available": False,
            "cuda_version": None,
            "cudnn_version": None,
            "gpu_name": None,
            "gpu_memory_total_gb": None,
            "gpu_capability": None,
            "gpu_multi_processor_count": None,
        })

    return info


HARDWARE_INFO = collect_hardware_info()

logger.info(f"Using device: {device}, compute type: {compute_type}")
logger.info(f"Model cache root: {MODEL_CACHE_ROOT}")
logger.info(f"Offline mode: {'ENABLED' if os.getenv('HF_HUB_OFFLINE') == '1' else 'DISABLED'} (HF_HUB_OFFLINE={os.getenv('HF_HUB_OFFLINE')})")
logger.info(
    "Hardware summary: "
    f"platform={HARDWARE_INFO['platform']}, "
    f"cpu={HARDWARE_INFO['processor']}, "
    f"cores={HARDWARE_INFO['cpu_cores_logical']}, "
    f"ram_gb={HARDWARE_INFO['ram_total_gb']}, "
    f"gpu={HARDWARE_INFO['gpu_name']}, "
    f"cuda_version={HARDWARE_INFO['cuda_version']}"
)