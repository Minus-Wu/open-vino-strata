"""
硬體檢測與 PolyStrata 35B-A3B 分層記憶體就緒度評估工具 (scripts/check_devices.py)
功能：
1. 掃描 OpenVINO 2026 可用硬體加速器 (CPU / GPU / NPU) 與支援之 Strata 屬性 (OFFLOAD_RATIO 等)
2. 檢測系統實體 RAM 與 GPU 顯存上限
3. 針對 Qwen3.8-35B-A3B (qwen35moe) 輸出各 VRAM 預算檔位 (6GB ~ 16GB) 的 Strata 專家卸載試算表
4. 列出本機目前已發現的 OpenVINO 模型與預估顯存佔用
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.engine.device_manager import device_manager, StrataConfig
from src.engine.ov_strata_engine import OVStrataEngine


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Check OpenVINO 2026 devices and PolyStrata readiness for Qwen3.8-35B-A3B."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen3.8-35B-A3B-int4-ov",
        help="Target model name or directory for Strata simulation (default: Qwen3.8-35B-A3B-int4-ov)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(" OpenVINO-Strata 2026 Hardware & Stratified MoE Readiness Report")
    print("=" * 80)

    # 1. 系統與裝置清單
    devices_list = device_manager.get_all_devices_summary()
    sys_mem = device_manager.get_system_memory_info()
    print(f"\n[System Host Memory (LPDDR5X / DDR5)]")
    print(
        f"  Total RAM     : {sys_mem['total_ram_gb']:.2f} GB\n"
        f"  Available RAM : {sys_mem['available_ram_gb']:.2f} GB\n"
        f"  Used RAM      : {sys_mem['used_ram_gb']:.2f} GB ({sys_mem['percent']}%)"
    )

    print(f"\n[OpenVINO Available Accelerators]: {', '.join(device_manager.available_devices)}")
    print("-" * 80)
    for info in devices_list:
        dev = info.get("device", "Unknown")
        print(f"  * [{dev}] {info.get('full_name', 'Unknown')}")
        if "total_memory_gb" in info:
            print(f"      Total Device Memory    : {info['total_memory_gb']} GB")
        if "max_alloc_memory_gb" in info:
            print(f"      Max Allocatable Memory : {info['max_alloc_memory_gb']} GB")
        if dev.startswith("GPU"):
            print(
                f"      Supports OFFLOAD_RATIO : {info.get('supports_offload_ratio', False)}"
            )
            print(f"      Execution Units        : {info.get('execution_units', 'N/A')}")
            print(f"      Capabilities           : {', '.join(info.get('capabilities', []))}")

    # 2. Qwen3.8-35B-A3B Strata 顯存預算試算矩陣
    print("\n" + "=" * 80)
    print(f" PolyStrata Memory Tiering Matrix for [{args.model}]")
    print(" Architecture: qwen35moe (35B Total, ~3B Active, 30 DeltaNet + 10 Full Attn, 256 Experts)")
    print("=" * 80)
    print(
        f"{'VRAM Budget':<14} | {'Offload Ratio':<14} | {'Est. VRAM (GPU)':<16} | "
        f"{'Est. RAM (CPU)':<16} | {'KV Cache':<8} | {'Status'}"
    )
    print("-" * 80)

    target_dev = "GPU" if device_manager.is_device_available("GPU") else "CPU"
    budgets = [6.0, 8.0, 10.0, 12.0, 14.5, 16.0]
    for b in budgets:
        cfg = StrataConfig(
            device=target_dev,
            vram_budget_gb=b,
            kv_cache_precision="u8",
            enable_mtp=True,
        )
        plan = device_manager.plan_strata_execution(args.model, cfg)
        fits_ram = plan.estimated_ram_offload_gb <= (sys_mem["available_ram_gb"] + 4.0)
        status = "READY (Optimal)" if fits_ram else "LOW HOST RAM"
        print(
            f"{b:>8.1f} GB   | {plan.recommended_offload_ratio:>11}%  | "
            f"{plan.estimated_vram_usage_gb:>12.2f} GB | "
            f"{plan.estimated_ram_offload_gb:>12.2f} GB | "
            f"{cfg.kv_cache_precision:>8} | {status}"
        )

    # 3. 本機自動硬體規劃建議
    auto_cfg = StrataConfig(device=target_dev)
    auto_plan = device_manager.plan_strata_execution(args.model, auto_cfg)
    print("\n[Auto-Detected Strata Plan for Current Machine]")
    for note in auto_plan.notes:
        print(f"  -> {note}")
    print(f"  -> OpenVINO Plugin Config: {auto_plan.ov_plugin_config}")

    # 4. 本機已安裝模型清單
    engine = OVStrataEngine(auto_load=False)
    models = engine.discover_available_models()
    print("\n" + "=" * 80)
    print(f" Discovered Local OpenVINO Models ({len(models)} found)")
    print("=" * 80)
    if not models:
        print("  (No local OpenVINO IR models found yet. Run `python scripts/download_model.py` to download.)")
    else:
        for m in models:
            print(
                f"  - {m['id']:<40} | Size: {m['total_weight_gb']:>5.2f} GB | "
                f"MoE: {str(m['is_moe']):<5} | VLM: {str(m['has_vision']):<5} | "
                f"MTP: {str(m['has_mtp']):<5} | Path: {m['path']}"
            )
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
