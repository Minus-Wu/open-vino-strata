"""
OpenVINO-Strata 模型轉換與混合精度量化工具 (scripts/convert_model.py)
功能：
將 HuggingFace Safetensors 模型 (如 `empero-ai/Qwen3.8-35B-A3B-Distill` 或 `Qwen/Qwen3.6-35B-A3B`)
轉換並量化為 OpenVINO 2026 INT4 + INT8 混合精度 IR 格式 (`openvino_language_model.xml` / `.bin`)，
使 35B-A3B MoE 模型能夠直接搭配 OpenVINO `OFFLOAD_RATIO`、`WEIGHTS_PATH` 與 `KV_CACHE_PRECISION="u8"` 進行低顯存分層推論。

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


def build_optimum_command(
    model_id: str,
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

    # Qwen3.8-35B-A3B-Distill / Qwen3.6-35B-A3B 為 Qwen3_5MoeForConditionalGeneration，需使用 image-text-to-text 導出拆分式架構
    resolved_task = task
    if resolved_task == "auto":
        if "35b-a3b" in model_id.lower() or "qwen3.8" in model_id.lower() or "qwen3.6" in model_id.lower():
            resolved_task = "image-text-to-text"
        else:
            resolved_task = "text-generation-with-past"

    cmd.extend([
        "--model", model_id,
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
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def install_conversion_dependencies() -> int:
    print("[Info] Installing PyTorch (CPU), transformers>=5.2.0, nncf, and optimum-intel (git main for qwen35moe)...")
    cmds = [
        [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu",
        ],
        [
            sys.executable, "-m", "pip", "install", "-U",
            "transformers>=5.2.0", "accelerate", "nncf",
            "git+https://github.com/huggingface/optimum-intel.git",
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
        help="Local HuggingFace download cache directory on D: drive (default: ./.hf_cache)",
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="Delete temporary .hf_cache (66 GB Safetensors) after successful OpenVINO IR export",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Automatically install torch, transformers>=5.2.0, optimum-intel, and nncf before exporting",
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
    cmd = build_optimum_command(
        model_id=args.model_id,
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
    print(f"  Target IR Path    : {out_dir}")
    print(f"  HF Cache Path     : {hf_cache_dir} (avoids C: drive space exhaustion)")
    print(f"  Quantization      : {args.weight_format.upper()} (group_size={args.group_size}, ratio={args.ratio}, sym={args.sym})")
    print(f"  Backup Precision  : {args.backup_precision} (protects MoE Router & Embeddings)")
    print(f"  Export Command    :\n    {' '.join(cmd)}")
    print("-" * 84)
    print("  [Requirement Note]")
    print("  - Source `empero-ai/Qwen3.8-35B-A3B-Distill` is 66 GB (21 BF16 Safetensors shards).")
    print("  - Exporting to OpenVINO INT4 IR produces an ~18 GB folder in `models/`.")
    print("  - Requires ~85 GB free disk space on D: (use `--clean-cache` to free the 66 GB after export)")
    print("    and Windows Virtual Memory (Pagefile) >= 64 GB.")
    print("=" * 84)

    if args.dry_run:
        print("\n[Dry-Run] Exiting without running conversion.")
        return 0

    if args.install_deps or not check_conversion_dependencies():
        if not args.install_deps:
            print("\n[Warning] Conversion dependencies (`torch`, `optimum-intel`, `nncf`) are not installed.")
            print("Run with `--install-deps` to install them automatically.")
            return 1
        rc = install_conversion_dependencies()
        if rc != 0:
            return rc

    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HOME"] = str(hf_cache_dir)
    env["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_dir / "hub")
    env["TMPDIR"] = str(hf_cache_dir / "tmp")
    env["TEMP"] = str(hf_cache_dir / "tmp")
    env["TMP"] = str(hf_cache_dir / "tmp")
    (hf_cache_dir / "tmp").mkdir(parents=True, exist_ok=True)

    res = subprocess.run(cmd, env=env)
    if res.returncode == 0:
        if args.clean_cache and hf_cache_dir.exists():
            print(f"\n[Clean-Cache] Removing temporary 66 GB HF cache at {hf_cache_dir} ...")
            shutil.rmtree(hf_cache_dir, ignore_errors=True)
        print("\n" + "=" * 84)
        print(f" Conversion Succeeded! OpenVINO IR saved to: {out_dir}")
        print(f" Run server: python run_server.py --model \"{out_dir.name}\"")
        print("=" * 84)
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
