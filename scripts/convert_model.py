"""
OpenVINO-Strata 模型轉換與混合精度量化工具 (scripts/convert_model.py)
功能：
1. 先透過 `curl.exe -L -C -` 斷點續傳將 HuggingFace Safetensors 模型 (如 `empero-ai/Qwen3.8-35B-A3B-Distill`)
   直接下載至本專案 `.hf_cache/<model_name>` (不透過 HF symlink cache，避免 Windows 空間翻倍或中斷重來)
2. 呼叫 `optimum-cli export openvino` 轉換並量化為 OpenVINO 2026 INT4 + INT8 混合精度 IR 格式 (`openvino_language_model.xml` / `.bin`)
3. 轉檔成功後可透過 `--clean-cache` 自動刪除 66 GB 暫存原始權重，僅保留約 18 GB 的 OpenVINO INT4 模型。

量化配方 (對標 OpenVINO 官方 Qwen3.6-35B-A3B-int4-ov 與 PolyStrata 分層策略)：
- Routed Experts (256 個專家 FFN): INT4 Asymmetric, group_size=64, ratio=1.0
- Router Gate / Embeddings / LM Head / Vision: INT8 Symmetric (`--backup-precision int8_sym`)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.download_model import download_repo


def build_optimum_command(
    model_source: str,
    output_dir: Path,
    weight_format: str = "int4",
    group_size: int = 64,
    ratio: float = 1.0,
    backup_precision: str = "int8_sym",
    sym: bool = False,
    task: str = "auto",
) -> List[str]:
    python_exe = Path(sys.executable)
    optimum_cli = python_exe.parent / ("optimum-cli.exe" if sys.platform == "win32" else "optimum-cli")

    if optimum_cli.exists():
        cmd = [str(optimum_cli), "export", "openvino"]
    else:
        cmd = [str(python_exe), "-m", "optimum.exporters.openvino"]

    resolved_task = task
    if resolved_task == "auto":
        if "35b-a3b" in model_source.lower() or "qwen3.8" in model_source.lower() or "qwen3.6" in model_source.lower():
            resolved_task = "image-text-to-text"
        else:
            resolved_task = "text-generation-with-past"

    cmd.extend([
        "--model", model_source,
        "--task", resolved_task,
        "--weight-format", weight_format,
        "--group-size", str(group_size),
        "--ratio", str(ratio),
        "--backup-precision", backup_precision,
        "--trust-remote-code",
    ])
    if sym:
        cmd.append("--sym")

    cmd.append(str(output_dir))
    return cmd


def check_conversion_dependencies() -> bool:
    try:
        import optimum.intel  # noqa: F401
        import nncf  # noqa: F401
        import torch  # noqa: F401
        import transformers
        return transformers.__version__.startswith("5.2.")
    except ImportError:
        return False


def install_conversion_dependencies() -> int:
    print("[Info] Installing PyTorch (CPU), transformers==5.2.*, nncf, and optimum-intel (git main for qwen35moe)...")
    cmds = [
        [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu",
        ],
        [
            sys.executable, "-m", "pip", "install", "-U",
            "accelerate", "nncf",
            "git+https://github.com/huggingface/optimum-intel.git",
        ],
        [
            sys.executable, "-m", "pip", "install",
            "transformers==5.2.*",
        ],
        [
            sys.executable, "-m", "pip", "install", "-U",
            "openvino==2026.4.1", "openvino-tokenizers==2026.4.1.0", "openvino-genai==2026.4.1.0",
        ],
    ]
    for c in cmds:
        print(f"  Running: {' '.join(c)}")
        rc = subprocess.run(c).returncode
        if rc != 0:
            return rc
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Convert Qwen3.8-35B-A3B-Distill / Qwen3.6-35B-A3B to OpenVINO INT4 mixed-precision IR."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="empero-ai/Qwen3.8-35B-A3B-Distill",
        help="HuggingFace model ID or local path (default: empero-ai/Qwen3.8-35B-A3B-Distill)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for OpenVINO IR (default: models/<model-name>-int4-ov)",
    )
    parser.add_argument(
        "--weight-format",
        type=str,
        default="int4",
        choices=["int4", "int8", "fp16"],
        help="Primary weight quantization format (default: int4)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64 for MoE accuracy)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Ratio of weights compressed to primary format (default: 1.0)",
    )
    parser.add_argument(
        "--backup-precision",
        type=str,
        default="int8_sym",
        choices=["int8_sym", "int8_asym", "none"],
        help="Backup precision for sensitive layers (Router / Embeddings / LM Head, default: int8_sym)",
    )
    parser.add_argument(
        "--sym",
        action="store_true",
        help="Use symmetric quantization for primary INT4 weights (default: asymmetric)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="auto",
        help="Export task (default: 'auto' -> 'image-text-to-text' for 35B-A3B qwen35moe)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(ROOT_DIR / ".hf_cache"),
        help="Local download staging directory on D: drive (default: ./.hf_cache)",
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="Delete temporary .hf_cache (66 GB Safetensors) after successful OpenVINO IR export",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Automatically install torch, transformers==5.2.*, optimum-intel, and nncf before exporting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the conversion command and memory requirements without executing",
    )
    args = parser.parse_args()

    model_short_name = Path(args.model_id).name
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = ROOT_DIR / "models" / f"{model_short_name}-{args.weight_format}-ov"

    hf_cache_dir = Path(args.cache_dir).resolve()
    is_local_path = Path(args.model_id).exists()
    local_source_dir = Path(args.model_id).resolve() if is_local_path else (hf_cache_dir / model_short_name)

    cmd = build_optimum_command(
        model_source=str(local_source_dir),
        output_dir=out_dir,
        weight_format=args.weight_format,
        group_size=args.group_size,
        ratio=args.ratio,
        backup_precision=args.backup_precision,
        sym=args.sym,
        task=args.task,
    )

    print("=" * 84)
    print(" OpenVINO-Strata Model Conversion & NNCF Quantization Utility")
    print("=" * 84)
    print(f"  Source Model      : {args.model_id}")
    print(f"  Staging Path      : {local_source_dir} (resumable curl.exe download on D:)")
    print(f"  Target IR Path    : {out_dir}")
    print(f"  Quantization      : {args.weight_format.upper()} (group_size={args.group_size}, ratio={args.ratio}, sym={args.sym})")
    print(f"  Backup Precision  : {args.backup_precision} (protects MoE Router & Embeddings)")
    print(f"  Export Command    :\n    {' '.join(cmd)}")
    print("-" * 84)
    print("  [Requirement Note]")
    print("  - Source `empero-ai/Qwen3.8-35B-A3B-Distill` is 66 GB (22 Safetensors files).")
    print("  - Exporting to OpenVINO INT4 IR produces an ~18 GB folder in `models/`.")
    print("  - Requires ~85 GB free disk space on D: (use `--clean-cache` to free the 66 GB after export)")
    print("    and Windows Virtual Memory (Pagefile) >= 64 GB.")
    print("=" * 84)

    if args.dry_run:
        print("\n[Dry-Run] Exiting without running conversion.")
        return 0

    if not check_conversion_dependencies():
        if args.install_deps:
            rc = install_conversion_dependencies()
            if rc != 0:
                return rc
        else:
            print("\n[Warning] Required conversion packages (`transformers==5.2.*`, `optimum-intel`, `nncf`) are missing or mismatched.")
            print("Re-run with `--install-deps` to fix automatically.")
            return 1

    # Step 1: 若來源為 HuggingFace Repo ID，先以 curl.exe 斷點續傳下載至 .hf_cache/<model_name>
    if not is_local_path:
        print(f"\n[Step 1/2] Downloading {args.model_id} to {local_source_dir} (resumable via curl.exe)...")
        dl_rc = download_repo(repo_id=args.model_id, target_dir=local_source_dir)
        if dl_rc != 0:
            print("[Error] Download interrupted or failed. Re-run the exact same command to resume.")
            return dl_rc

    # Step 2: 執行 OpenVINO INT4 混合精度量化導出
    print(f"\n[Step 2/2] Exporting and quantizing {local_source_dir} -> {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = hf_cache_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HOME"] = str(hf_cache_dir)
    env["TMPDIR"] = str(tmp_dir)
    env["TEMP"] = str(tmp_dir)
    env["TMP"] = str(tmp_dir)

    res = subprocess.run(cmd, env=env)
    if res.returncode == 0:
        if args.clean_cache and not is_local_path and local_source_dir.exists():
            print(f"\n[Clean-Cache] Removing temporary 66 GB source files at {local_source_dir} ...")
            shutil.rmtree(local_source_dir, ignore_errors=True)
        print("\n" + "=" * 84)
        print(f" Conversion Succeeded! OpenVINO IR saved to: {out_dir}")
        print(f" Run server: python run_server.py --model \"{out_dir.name}\"")
        print("=" * 84)
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
