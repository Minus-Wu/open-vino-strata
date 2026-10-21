"""
OpenVINO-Strata 效能與顯存分層基準測試工具 (scripts/benchmark.py)
量測指標：
- 模型編譯與載入時間 (Load Time)
- 首字延遲 (TTFT, ms)
- 預填充吞吐量 (Prefill TPS, tok/s)
- 解碼生成吞吐量 (Decode TPS, tok/s)
- 端到端整體吞吐量 (End-to-End TPS, tok/s)
- GPU 顯存分配量 (VRAM MB) 與系統記憶體使用量 (RAM GB)
- 支援 `--sweep-offload` 一次性對比不同專家卸載比例 (OFFLOAD_RATIO) 對顯存與吞吐量的影響
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.engine.device_manager import StrataConfig
from src.engine.ov_strata_engine import OVStrataEngine


async def run_single_benchmark(
    model_path: str,
    device: str,
    offload_ratio: int,
    kv_precision: str,
    enable_mtp: bool,
    prompt: str,
    max_tokens: int,
    runs: int,
) -> Dict[str, Any]:
    cfg = StrataConfig(
        device=device,
        offload_ratio=offload_ratio,
        kv_cache_precision=kv_precision,
        enable_mtp=enable_mtp,
    )
    engine = OVStrataEngine(
        model_path=model_path,
        device=device,
        strata_config=cfg,
        auto_load=True,
    )

    # Warmup run
    formatted_prompt, _, _ = engine.format_chat_prompt(
        [{"role": "user", "content": "Hello!"}],
        enable_thinking=False,
    )
    await engine.generate(formatted_prompt, max_new_tokens=8, temperature=0.0, do_sample=False)

    # Timed benchmark runs
    bench_prompt, _, _ = engine.format_chat_prompt(
        [{"role": "user", "content": prompt}],
        enable_thinking=False,
    )

    ttft_list: List[float] = []
    prefill_tps_list: List[float] = []
    decode_tps_list: List[float] = []
    total_tps_list: List[float] = []
    vram_mb = 0.0
    ram_gb = 0.0
    p_toks = 0
    c_toks = 0

    for _ in range(runs):
        _, metrics = await engine.generate(
            bench_prompt,
            max_new_tokens=max_tokens,
            temperature=0.0,
            do_sample=False,
        )
        m = metrics.to_dict()
        ttft_list.append(m["ttft_seconds"] * 1000.0)
        prefill_tps_list.append(m["prefill_tokens_per_second"])
        decode_tps_list.append(m["decode_tokens_per_second"])
        total_tps_list.append(m["tokens_per_second"])
        vram_mb = max(vram_mb, m["vram_allocated_mb"])
        ram_gb = max(ram_gb, m["ram_used_gb"])
        p_toks = metrics.prompt_tokens
        c_toks = metrics.completion_tokens

    load_sec = engine._load_duration
    actual_offload = engine.current_plan.recommended_offload_ratio if engine.current_plan else offload_ratio
    model_name = engine.model_name
    await engine.unload_model()

    avg = lambda xs: sum(xs) / max(1, len(xs))
    return {
        "model": model_name,
        "device": device,
        "offload_ratio": actual_offload,
        "kv_precision": kv_precision,
        "load_s": round(load_sec, 2),
        "prompt_tokens": p_toks,
        "completion_tokens": c_toks,
        "ttft_ms": round(avg(ttft_list), 1),
        "prefill_tps": round(avg(prefill_tps_list), 1),
        "decode_tps": round(avg(decode_tps_list), 2),
        "total_tps": round(avg(total_tps_list), 2),
        "vram_mb": round(vram_mb, 1),
        "ram_gb": round(ram_gb, 2),
    }


async def async_main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="OpenVINO-Strata LLM & MoE Offload Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path (defaults to first discovered local model)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        help="Target device: GPU, CPU, NPU (default: GPU)",
    )
    parser.add_argument(
        "--offload-ratio",
        type=int,
        default=-1,
        help="Expert offload ratio 0..100 (-1 for auto Strata plan)",
    )
    parser.add_argument(
        "--sweep-offload",
        type=str,
        default=None,
        help="Comma-separated offload ratios to compare, e.g. '0,50,85'",
    )
    parser.add_argument(
        "--kv-precision",
        type=str,
        default="u8",
        choices=["u8", "f16"],
        help="KV cache precision (default: u8)",
    )
    parser.add_argument(
        "--enable-mtp",
        action="store_true",
        help="Enable Multi-Token Prediction speculative decoding if supported by model",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max new tokens to generate per run (default: 64)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of timed runs per configuration (default: 2)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a clean Python function to implement binary search on a sorted list and explain its time complexity.",
        help="Benchmark prompt text",
    )
    args = parser.parse_args()

    discovery_engine = OVStrataEngine(auto_load=False)
    discovered = discovery_engine.discover_available_models()
    if args.model:
        target_model = str(discovery_engine.resolve_model_path(args.model))
    elif discovered:
        target_model = discovered[0]["path"]
    else:
        print("[Error] No local OpenVINO models found. Run `python scripts/download_model.py` first.")
        return 1

    ratios: List[int] = (
        [int(x.strip()) for x in args.sweep_offload.split(",") if x.strip()]
        if args.sweep_offload
        else [args.offload_ratio]
    )

    print("=" * 94)
    print(" OpenVINO-Strata Performance & Tiered Memory Benchmark")
    print("=" * 94)
    print(f"  Target Model : {target_model}")
    print(f"  Device       : {args.device} | KV Cache: {args.kv_precision} | Max Tokens: {args.max_tokens} | Runs: {args.runs}")
    print("-" * 94)

    results: List[Dict[str, Any]] = []
    for r in ratios:
        label = "Auto" if r < 0 else f"{r}%"
        print(f"-> Running benchmark with OFFLOAD_RATIO={label} ...")
        res = await run_single_benchmark(
            model_path=target_model,
            device=args.device,
            offload_ratio=r,
            kv_precision=args.kv_precision,
            enable_mtp=args.enable_mtp,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            runs=args.runs,
        )
        results.append(res)

    print("\n" + "=" * 94)
    print(
        f"{'Model':<34} | {'Dev':<4} | {'Offload':<7} | {'Load(s)':<7} | "
        f"{'TTFT(ms)':<8} | {'Decode(t/s)':<11} | {'VRAM(MB)':<9} | {'RAM(GB)'}"
    )
    print("-" * 94)
    for r in results:
        print(
            f"{r['model'][:34]:<34} | {r['device']:<4} | {r['offload_ratio']:>5}%  | "
            f"{r['load_s']:>7.2f} | {r['ttft_ms']:>8.1f} | {r['decode_tps']:>11.2f} | "
            f"{r['vram_mb']:>9.1f} | {r['ram_gb']:>6.2f}"
        )
    print("=" * 94)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(async_main()))
