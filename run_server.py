"""
OpenVINO-Strata LLM Serving 服務啟動入口
提供命令列參數配置，支援 Qwen3.8-35B-A3B / Qwen3.6-35B-A3B 等大型稀疏 MoE 模型之
低顯存分層推論 (OFFLOAD_RATIO + KV INT8 壓縮 + MTP 投機解碼)，並暴露 OpenAI / Anthropic 相容 API。
"""

import argparse
import io
import logging
import sys
from pathlib import Path

import uvicorn

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

project_root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.paths import get_app_root
project_root = get_app_root()

from src.engine.device_manager import device_manager, StrataConfig
from src.engine.ov_strata_engine import is_valid_ov_model_dir
from src.server.app import create_app

import logging.handlers

logs_dir = project_root / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
server_log_file = logs_dir / "server.log"

log_file_handler = logging.handlers.RotatingFileHandler(
    server_log_file,
    maxBytes=20 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
log_file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        log_file_handler,
    ],
)
logging.getLogger("uvicorn").addHandler(log_file_handler)
logging.getLogger("uvicorn.access").addHandler(log_file_handler)
logger = logging.getLogger("ov_strata")


def resolve_initial_model(model_arg: str | None) -> Path | None:
    """解析啟動時欲載入的模型目錄 (優先順序：命令列參數 -> 本專案 models/ -> 隔壁 open-vino-llm/models/)"""
    search_dirs = [
        project_root / "models",
        project_root.parent.parent / "aio-mono" / "open-vino-llm" / "models",
    ]

    if model_arg:
        p = Path(model_arg)
        if p.exists() and is_valid_ov_model_dir(p):
            return p.resolve()
        for base in search_dirs:
            cand = base / model_arg
            if cand.exists() and is_valid_ov_model_dir(cand):
                return cand.resolve()
            if base.exists():
                for sub in sorted(base.iterdir()):
                    if is_valid_ov_model_dir(sub) and model_arg.lower() in sub.name.lower():
                        return sub.resolve()
        return None

    # 未指定時，優先挑選 35B-A3B 模型，若無則選第一個可用模型
    all_found: list[Path] = []
    for base in search_dirs:
        if not base.exists():
            continue
        for sub in sorted(base.iterdir()):
            if is_valid_ov_model_dir(sub):
                all_found.append(sub.resolve())

    if not all_found:
        return None

    # 優先選擇 Qwen3.8-35B-A3B，其次為 Qwen3.6 / 其他 35B-A3B 模型
    for m in all_found:
        if "qwen3.8" in m.name.lower():
            return m
    for m in all_found:
        if "35b-a3b" in m.name.lower() or "qwen3.6" in m.name.lower():
            return m
    return all_found[0]


def resolve_sub_model(
    sub_model_arg: str | None,
    primary_path: Path | None,
    auto_sub: bool = True,
) -> Path | None:
    """解析副引擎小模型 (用於處理 Claude Code Haiku/標題/快回請求)"""
    search_dirs = [
        project_root / "models",
        project_root.parent.parent / "aio-mono" / "open-vino-llm" / "models",
    ]

    if sub_model_arg:
        p = Path(sub_model_arg)
        if p.exists() and is_valid_ov_model_dir(p):
            return p.resolve()
        for base in search_dirs:
            cand = base / sub_model_arg
            if cand.exists() and is_valid_ov_model_dir(cand):
                return cand.resolve()
            if base.exists():
                for sub in sorted(base.iterdir()):
                    if is_valid_ov_model_dir(sub) and sub_model_arg.lower() in sub.name.lower():
                        return sub.resolve()
        return None

    if not auto_sub:
        return None

    # 自動在模型目錄中尋找輕量小模型 (< 2.5 GB) 作為副引擎
    candidates: list[tuple[float, Path]] = []
    primary_resolved = primary_path.resolve() if primary_path else None

    for base in search_dirs:
        if not base.exists():
            continue
        for sub in sorted(base.iterdir()):
            if not is_valid_ov_model_dir(sub):
                continue
            if primary_resolved and sub.resolve() == primary_resolved:
                continue
            info = device_manager.inspect_model_dir(sub)
            if info.total_weight_gb < 2.5 and not info.is_moe:
                candidates.append((info.total_weight_gb, sub.resolve()))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    return None


def main():
    parser = argparse.ArgumentParser(
        description="OpenVINO-Strata: Low-VRAM MoE OpenAI/Anthropic-Compatible LLM Provider"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="監聽主機位址 (預設 127.0.0.1)")
    parser.add_argument("--port", type=int, default=12223, help="服務監聽連接埠 (預設 12223)")
    parser.add_argument("--model_dir", "--model", dest="model_dir", type=str, default=None, help="模型目錄名稱或完整路徑")
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        choices=["GPU", "NPU", "CPU", "AUTO", "HETERO:GPU,CPU"],
        help="目標運算硬體裝置 (預設 GPU)",
    )
    parser.add_argument(
        "--offload_ratio",
        "--offload-ratio",
        dest="offload_ratio",
        type=int,
        default=None,
        help="MoE 專家權重卸載至系統 RAM 的比例 (0~100%%，預設 None 由 Strata 自動依顯存預算計算)",
    )
    parser.add_argument(
        "--vram_budget_gb",
        "--vram-budget",
        dest="vram_budget_gb",
        type=float,
        default=None,
        help="最大 GPU 顯存預算 (GB)，例如設定 8.0 或 12.0 以模擬/限制低顯存環境",
    )
    parser.add_argument(
        "--vram_reserve_mib",
        type=int,
        default=2048,
        help="預留給系統與驅動的顯存安全底線 (MiB，預設 2048)",
    )
    parser.add_argument(
        "--kv_cache_precision",
        type=str,
        default="u8",
        choices=["u8", "f16"],
        help="KV Cache 壓縮精度 (預設 u8 即 INT8 壓縮，大幅降低長上下文顯存)",
    )
    parser.add_argument(
        "--enable_mtp",
        action="store_true",
        default=False,
        help="啟用內建 MTP (Multi-Token Prediction) 投機解碼草稿模型 (若模型具備 openvino_mtp_model.xml)",
    )
    parser.add_argument(
        "--mtp_device",
        type=str,
        default="CPU",
        choices=["CPU", "NPU", "GPU"],
        help="MTP 草稿模型執行裝置 (預設 CPU)",
    )
    parser.add_argument(
        "--idle_unload_seconds",
        type=int,
        default=0,
        help="閒置超過指定秒數自動卸載模型釋放顯存 (預設 0 常駐)",
    )
    parser.add_argument(
        "--sub_model",
        "--sub-model",
        dest="sub_model",
        type=str,
        default=None,
        help="副模型 (小模型/Fast Model) 目錄名稱或路徑，用於處理 Claude Code Haiku/標題/快回請求",
    )
    parser.add_argument(
        "--sub_device",
        "--sub-device",
        dest="sub_device",
        type=str,
        default="NPU",
        choices=["NPU", "CPU", "GPU"],
        help="副模型運算硬體裝置 (預設 NPU，不相容時自動降級至 CPU)",
    )
    parser.add_argument(
        "--auto_sub_engine",
        "--auto-sub-engine",
        dest="auto_sub_engine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在未指定 --sub-model 時自動在模型庫中挑選輕量模型 (<2.5GB) 啟用副引擎",
    )
    parser.add_argument(
        "--lazy_load",
        action="store_true",
        default=False,
        help="啟動時暫不載入模型至顯存，待收到首筆請求或 /api/models/load 時才載入",
    )

    args = parser.parse_args()
    model_path = resolve_initial_model(args.model_dir)

    # 地雷防護：檢查是否嘗試在 NPU 上運行 35B MoE 主模型
    if args.device.upper() == "NPU":
        inspect_res = device_manager.inspect_model_dir(model_path) if model_path else None
        if inspect_res and (inspect_res.is_moe or "35b" in inspect_res.name.lower()):
            raise ValueError(
                f"裝置設定不支援：模型 [{model_path.name if model_path else 'MoE'}] 包含稀疏專家架構，"
                "OpenVINO NPU 驅動不支援 OFFLOAD_RATIO 分層卸載機制。\n"
                "請將主模型 --device 指定為 GPU (推薦) 或 CPU。"
                "若需善用 NPU 算力，請使用 --sub-device NPU --sub-model 搭配小型模型 (如 0.5B/1.7B) 作為副引擎。"
            )

    if model_path is None and not args.lazy_load:
        logger.warning(
            "目前在 models/ 目錄下尚未找到已下載的 OpenVINO 模型，將以 Lazy Mode 啟動服務。"
            "請執行 python scripts/download_model.py 或透過 /api/models/load 載入模型。"
        )
        args.lazy_load = True

    sub_model_path = resolve_sub_model(args.sub_model, model_path, auto_sub=args.auto_sub_engine)
    sub_ram_gb = 0.0
    if sub_model_path:
        sub_info = device_manager.inspect_model_dir(sub_model_path)
        sub_ram_gb = round(sub_info.total_weight_gb + 0.5, 2)

    strata_cfg = StrataConfig(
        device=args.device,
        offload_ratio=args.offload_ratio,
        vram_budget_gb=args.vram_budget_gb,
        vram_reserve_mib=args.vram_reserve_mib,
        kv_cache_precision=args.kv_cache_precision,
        enable_mtp=args.enable_mtp,
        mtp_device=args.mtp_device,
        idle_unload_seconds=args.idle_unload_seconds,
        sub_model_ram_budget_gb=sub_ram_gb,
    )

    print("=" * 72)
    print("   OpenVINO-Strata: Low-VRAM MoE LLM Provider (OpenAI & Anthropic API)")
    print("=" * 72)
    print(f"[*] 主模型 (Primary): {model_path.name if model_path else '(Lazy Mode 待命)'}")
    print(f"[*] 主裝置: [{args.device}] | KV 精度: [{args.kv_cache_precision}]")
    if sub_model_path:
        print(f"[*] 副引擎 (Sub-Engine): {sub_model_path.name} 於 [{args.sub_device}] (預留 RAM: {sub_ram_gb} GB)")
    else:
        print("[*] 副引擎 (Sub-Engine): (未啟用)")
    if model_path:
        plan = device_manager.plan_strata_execution(model_path, strata_cfg)
        print(
            f"[*] Strata 顯存規劃: 建議 OFFLOAD_RATIO={plan.recommended_offload_ratio}% "
            f"(預計 VRAM: {plan.estimated_vram_usage_gb} GB / 預算 {plan.vram_budget_gb} GB, "
            f"預計 RAM: {plan.estimated_ram_offload_gb} GB)"
        )
    print(f"[*] 服務端點: http://{args.host}:{args.port}")
    print(f"[*] OpenAI 端點: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"[*] Anthropic 端點: http://{args.host}:{args.port}/v1/messages")
    print(f"[*] Strata 管理端點: http://{args.host}:{args.port}/api/models")
    print(f"[*] 互動式 API 文檔: http://{args.host}:{args.port}/docs")
    print(f"[*] 可用硬體清單: {device_manager.available_devices}")
    print("=" * 72)

    app = create_app(
        model_path=model_path,
        device=args.device,
        strata_config=strata_cfg,
        lazy_load=args.lazy_load,
        sub_model_path=sub_model_path,
        sub_device=args.sub_device,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
