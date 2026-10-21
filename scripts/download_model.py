"""
OpenVINO-Strata 模型下載與斷點續傳工具 (scripts/download_model.py)
支援：
1. 官方預量化 OpenVINO IR 模型 (`OpenVINO/Qwen3.6-35B-A3B-int4-ov` 等)
2. 社群 Qwen3.8 蒸餾版 (`empero-ai/Qwen3.8-35B-A3B-Distill-GGUF` / `MTP-GGUF`)
3. 輕量驗證模型 (`Qwen2.5-Coder-0.5B-Instruct-int4-ov` / `3B`)
4. 採用 Windows 內建 `curl.exe -L -C -` 直接串流寫入專案 `models/`，支援斷點續傳且不佔用 C 槽快取。
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import quote

from huggingface_hub import HfApi

ROOT_DIR = Path(__file__).resolve().parent.parent

PRESET_MODELS: Dict[str, Dict[str, Any]] = {
    "qwen3.6-35b-a3b-int4-ov": {
        "repo_id": "OpenVINO/Qwen3.6-35B-A3B-int4-ov",
        "folder": "Qwen3.6-35B-A3B-int4-ov",
        "size_gb": 18.0,
        "desc": "Official OpenVINO INT4 35B-A3B MoE (qwen35moe, includes MTP & Vision submodels, ideal for Strata OFFLOAD_RATIO)",
    },
    "qwen3.8-35b-a3b-distill-gguf-q3": {
        "repo_id": "empero-ai/Qwen3.8-35B-A3B-Distill-GGUF",
        "folder": "Qwen3.8-35B-A3B-Distill-GGUF-Q3_K_M",
        "allow_patterns": ["*Q3_K_M.gguf", "*.json", "*.md"],
        "size_gb": 16.8,
        "desc": "Qwen3.8-35B-A3B-Distill Q3_K_M GGUF (~16.8 GiB, distilled Qwen3.8 reasoning on qwen35moe)",
    },
    "qwen3.8-35b-a3b-distill-gguf-iq2": {
        "repo_id": "empero-ai/Qwen3.8-35B-A3B-Distill-GGUF",
        "folder": "Qwen3.8-35B-A3B-Distill-GGUF-IQ2_M",
        "allow_patterns": ["*IQ2_M.gguf", "*.json", "*.md"],
        "size_gb": 11.7,
        "desc": "Qwen3.8-35B-A3B-Distill IQ2_M GGUF (~11.7 GiB, ultra-compact footprint)",
    },
    "qwen3.8-35b-a3b-distill-gguf-q4": {
        "repo_id": "empero-ai/Qwen3.8-35B-A3B-Distill-GGUF",
        "folder": "Qwen3.8-35B-A3B-Distill-GGUF-Q4_K_M",
        "allow_patterns": ["*Q4_K_M.gguf", "*.json", "*.md"],
        "size_gb": 20.7,
        "desc": "Qwen3.8-35B-A3B-Distill Q4_K_M GGUF (~20.7 GiB, high quality 4-bit)",
    },
    "qwen3.8-35b-a3b-distill-mtp-gguf": {
        "repo_id": "empero-ai/Qwen3.8-35B-A3B-Distill-MTP-GGUF",
        "folder": "Qwen3.8-35B-A3B-Distill-MTP-GGUF",
        "allow_patterns": ["*Q4_K_M.gguf", "*.json", "*.md"],
        "size_gb": 19.7,
        "desc": "Qwen3.8-35B-A3B-Distill with Multi-Token Prediction (MTP) heads (~19.7 GiB)",
    },
    "minicpm-v-4.7-35b-a3b-int4-ov": {
        "repo_id": "OpenVINO/MiniCPM-V-4_7-35B-A3B-int4-ov",
        "folder": "MiniCPM-V-4_7-35B-A3B-int4-ov",
        "size_gb": 18.2,
        "desc": "OpenVINO INT4 MiniCPM-V 4.7 (Qwen3.6-35B-A3B MoE backbone + Vision)",
    },
    "qwen3-8b-int4-ov": {
        "repo_id": "OpenVINO/Qwen3-8B-int4-ov",
        "folder": "Qwen3-8B-int4-ov",
        "size_gb": 4.8,
        "desc": "OpenVINO INT4 Qwen3-8B (Dense 8B with <think> reasoning & tool calling)",
    },
    "qwen2.5-coder-3b": {
        "repo_id": "OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov",
        "folder": "Qwen2.5-Coder-3B-Instruct-int4-ov",
        "size_gb": 1.9,
        "desc": "OpenVINO INT4 Qwen2.5-Coder-3B-Instruct (Fast 3B coding model)",
    },
    "qwen2.5-coder-0.5b": {
        "repo_id": "OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov",
        "folder": "Qwen2.5-Coder-0.5B-Instruct-int4-ov",
        "size_gb": 0.35,
        "desc": "OpenVINO INT4 Qwen2.5-Coder-0.5B-Instruct (Ultra-light POC verification model)",
    },
}


def _matches_patterns(filename: str, patterns: Optional[List[str]]) -> bool:
    if not patterns:
        return True
    from fnmatch import fnmatch
    return any(fnmatch(filename, pat) for pat in patterns)


def download_with_curl(url: str, dest_path: Path) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    curl_bin = shutil.which("curl.exe") or shutil.which("curl")
    if not curl_bin:
        return False

    cmd = [
        curl_bin,
        "-L",
        "-C", "-",
        "--retry", "5",
        "--retry-delay", "3",
        "--fail",
        "-o", str(dest_path),
        url,
    ]
    res = subprocess.run(cmd)
    return res.returncode == 0


def download_repo(
    repo_id: str,
    target_dir: Path,
    allow_patterns: Optional[List[str]] = None,
    skip_vision: bool = False,
    skip_mtp: bool = False,
) -> int:
    print("=" * 76)
    print(f" Downloading Model Repository : {repo_id}")
    print(f" Target Local Directory       : {target_dir}")
    print("=" * 76)

    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id)

    filtered_files: List[str] = []
    for f in files:
        if f.startswith(".git"):
            continue
        if not _matches_patterns(f, allow_patterns):
            continue
        if skip_vision and "vision" in f.lower():
            print(f"  [Skip Vision] Skipping {f}")
            continue
        if skip_mtp and "mtp" in f.lower():
            print(f"  [Skip MTP]    Skipping {f}")
            continue
        filtered_files.append(f)

    print(f"Total files to download: {len(filtered_files)}\n")
    target_dir.mkdir(parents=True, exist_ok=True)

    for idx, rel_file in enumerate(filtered_files, start=1):
        dest_file = target_dir / rel_file
        encoded_path = "/".join(quote(part) for part in rel_file.split("/"))
        url = f"https://huggingface.co/{repo_id}/resolve/main/{encoded_path}?download=true"
        print(f"[{idx}/{len(filtered_files)}] Downloading {rel_file} ...")
        ok = download_with_curl(url, dest_file)
        if not ok:
            print(f"\n[ERROR] Failed to download {rel_file}. Re-run the same command to resume.")
            return 1

    print("\n" + "=" * 76)
    print(f" Download Complete! Model ready at: {target_dir}")
    print(f" Start server with: python run_server.py --model \"{target_dir.name}\" --vram-budget 10.0")
    print("=" * 76)
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Download OpenVINO IR or GGUF 35B-A3B models with resume support."
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="qwen3.6-35b-a3b-int4-ov",
        choices=list(PRESET_MODELS.keys()),
        help="Preset model key to download (default: qwen3.6-35b-a3b-int4-ov)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Custom HuggingFace repo ID (overrides --preset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "models"),
        help="Base models directory (default: ./models)",
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip downloading openvino_vision_embeddings* files to save ~415 MB",
    )
    parser.add_argument(
        "--skip-mtp",
        action="store_true",
        help="Skip downloading openvino_mtp_model* files to save ~412 MB",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all preset models and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("=" * 88)
        print(" Available Preset Models for OpenVINO-Strata")
        print("=" * 88)
        for key, info in PRESET_MODELS.items():
            print(f"  * {key:<34} (~{info['size_gb']:>5.1f} GB) -> {info['repo_id']}")
            print(f"      {info['desc']}")
        print("=" * 88)
        print("\nNote: For converting `empero-ai/Qwen3.8-35B-A3B-Distill` (Safetensors) directly into")
        print("OpenVINO INT4 IR format (`openvino_language_model.xml`), use `python scripts/convert_model.py`.")
        return 0

    base_dir = Path(args.output_dir)
    if args.repo_id:
        folder_name = args.repo_id.split("/")[-1]
        return download_repo(
            repo_id=args.repo_id,
            target_dir=base_dir / folder_name,
            skip_vision=args.skip_vision,
            skip_mtp=args.skip_mtp,
        )

    preset_info = PRESET_MODELS[args.preset]
    return download_repo(
        repo_id=preset_info["repo_id"],
        target_dir=base_dir / preset_info["folder"],
        allow_patterns=preset_info.get("allow_patterns"),
        skip_vision=args.skip_vision,
        skip_mtp=args.skip_mtp,
    )


if __name__ == "__main__":
    sys.exit(main())
