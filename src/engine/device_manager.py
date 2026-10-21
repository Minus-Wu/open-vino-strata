"""
裝置管理與 PolyStrata 分層顯存規劃模組 (Device Manager & Strata Memory Planner)
提供 OpenVINO 運算硬體 (CPU / GPU / NPU) 查詢、即時 VRAM/RAM 監控，
以及針對 Qwen3.8-35B-A3B / Qwen3.6-35B-A3B 等稀疏 MoE 模型的自動分層卸載 (OFFLOAD_RATIO) 規劃。
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import openvino as ov

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("ov_strata.device_manager")


@dataclass
class StrataConfig:
    """PolyStrata 風格的分層推論配置"""
    device: str = "GPU"
    # None 或 -1 代表自動依照模型大小與目標 VRAM 預算計算最佳專家卸載比例 (0~100)
    offload_ratio: Optional[int] = None
    # 目標最大顯存預算 (GB)，例如在 8GB / 12GB / 16GB 顯卡上限制顯存用量；None 代表自動保留安全餘裕
    vram_budget_gb: Optional[float] = None
    # 顯存保留安全底線 (MiB)，避免 Intel xe / Windows 共享顯存分配過滿導致驅動卡死
    vram_reserve_mib: int = 2048
    # KV Cache 精度: "u8" (INT8 壓縮，省顯存首選) 或 "f16"
    kv_cache_precision: str = "u8"
    # Scheduler KV Cache 上限 (GB)
    kv_cache_size_gb: Optional[float] = None
    # 是否啟用對話前綴快取 (Conversation Prefix Caching)
    enable_prefix_caching: bool = True
    # 是否啟用長上下文 KV Cache 自動淘汰壓縮 (Cache Eviction / KVCrush)
    enable_cache_eviction: bool = False
    # 是否啟用內建 MTP (Multi-Token Prediction) 投機解碼草稿模型
    enable_mtp: bool = False
    # MTP 草稿模型運算裝置 ("NPU", "CPU", "GPU")
    mtp_device: str = "CPU"
    # 每次投機解碼預測的草稿 token 數
    num_assistant_tokens: int = 3
    # 是否啟用多模態圖片輸入 (Vision)
    enable_vision: bool = True
    # 閒置自動釋放顯存秒數 (0 代表常駐不自動卸載)
    idle_unload_seconds: int = 0
    # 是否啟用編譯快取 (.ov_cache) 加速二次啟動
    enable_compile_cache: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelInspectResult:
    """模型目錄結構與權重分析結果"""
    name: str
    path: str
    format: str = "openvino_ir"          # "openvino_ir", "openvino_vlm_ir", "gguf", "unknown"
    architecture: str = "unknown"        # 例如 "qwen3_5_moe", "qwen3_moe", "qwen35moe", "qwen2"
    is_moe: bool = False
    has_linear_attention: bool = False
    has_mtp: bool = False
    has_vision: bool = False
    total_weight_gb: float = 0.0
    language_weight_gb: float = 0.0
    embedding_weight_gb: float = 0.0
    mtp_weight_gb: float = 0.0
    vision_weight_gb: float = 0.0
    estimated_dense_gb: float = 0.0      # 常駐顯存之 Dense + Shared Expert + Router 權重估算
    estimated_routed_experts_gb: float = 0.0  # 可分層卸載之 Routed Experts 權重估算
    num_layers: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 0
    quantization: str = "INT4"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrataPlan:
    """Strata 分層顯存規劃報告"""
    model_info: ModelInspectResult
    target_device: str
    vram_total_gb: float
    vram_budget_gb: float
    ram_total_gb: float
    ram_available_gb: float
    recommended_offload_ratio: int       # 0 ~ 100 (%)
    kv_cache_precision: str
    kv_cache_budget_gb: float
    estimated_vram_usage_gb: float       # 預計占用 GPU 顯存 (Dense + Hot Experts + KV Cache)
    estimated_ram_offload_gb: float      # 預計卸載至系統 RAM 之 Cold Experts 與快取
    hot_experts_in_vram_pct: int         # 駐留於 GPU VRAM 的熱點專家比例 (%)
    cold_experts_in_ram_pct: int         # 卸載至系統 RAM/CPU 的專家比例 (%)
    use_mtp: bool
    mtp_device: str
    use_prefix_caching: bool
    ov_plugin_config: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class DeviceManager:
    def __init__(self):
        self._core = ov.Core()
        self._available_devices = self._core.available_devices

    @property
    def core(self) -> ov.Core:
        return self._core

    @property
    def available_devices(self) -> List[str]:
        self._available_devices = self._core.available_devices
        return self._available_devices

    def is_device_available(self, device_name: str) -> bool:
        device_upper = device_name.upper()
        if device_upper.startswith(("HETERO:", "AUTO:", "MULTI:")):
            sub_devs = device_upper.split(":", 1)[1].split(",")
            return all(self.is_device_available(d.strip()) for d in sub_devs if d.strip())
        return any(
            d.upper() == device_upper or device_upper.startswith(d.upper())
            for d in self.available_devices
        )

    def get_system_memory_info(self) -> Dict[str, float]:
        """取得系統實體記憶體 (RAM) 總量與可用量 (GB)"""
        if psutil is not None:
            vm = psutil.virtual_memory()
            return {
                "total_ram_gb": round(vm.total / (1024 ** 3), 2),
                "available_ram_gb": round(vm.available / (1024 ** 3), 2),
                "used_ram_gb": round((vm.total - vm.available) / (1024 ** 3), 2),
                "percent": vm.percent,
            }
        return {"total_ram_gb": 32.0, "available_ram_gb": 20.0, "used_ram_gb": 12.0, "percent": 37.5}

    def get_gpu_memory_stats(self, device_name: str = "GPU") -> Dict[str, Any]:
        """查詢 Intel GPU 總顯存、最大單一分配顯存與目前即時統計"""
        matched = [
            d for d in self.available_devices
            if d.upper() == device_name.upper() or device_name.upper().startswith(d.upper())
        ]
        dev = matched[0] if matched else "GPU"
        stats: Dict[str, Any] = {"device": dev, "total_memory_gb": 0.0, "max_alloc_memory_gb": 0.0}
        if "GPU" not in dev.upper():
            return stats

        try:
            mem_bytes = self._core.get_property(dev, "GPU_DEVICE_TOTAL_MEM_SIZE")
            stats["total_memory_gb"] = round(mem_bytes / (1024 ** 3), 2)
        except Exception:
            pass

        try:
            max_alloc = self._core.get_property(dev, "GPU_DEVICE_MAX_ALLOC_MEM_SIZE")
            stats["max_alloc_memory_gb"] = round(max_alloc / (1024 ** 3), 2)
        except Exception:
            stats["max_alloc_memory_gb"] = stats["total_memory_gb"]

        try:
            raw_mem_stats = self._core.get_property(dev, "GPU_MEMORY_STATISTICS")
            if isinstance(raw_mem_stats, dict):
                stats["memory_statistics"] = raw_mem_stats
                used_bytes = sum(v for v in raw_mem_stats.values() if isinstance(v, (int, float)))
                stats["allocated_memory_mb"] = round(used_bytes / (1024 ** 2), 2)
        except Exception:
            pass

        return stats

    def supports_offload_ratio(self, device_name: str = "GPU") -> bool:
        """檢查指定裝置是否支援 OpenVINO 2026 MoE OFFLOAD_RATIO 屬性"""
        matched = [
            d for d in self.available_devices
            if d.upper() == device_name.upper() or device_name.upper().startswith(d.upper())
        ]
        if not matched:
            return False
        dev = matched[0]
        try:
            supported = self._core.get_property(dev, "SUPPORTED_PROPERTIES")
            return "OFFLOAD_RATIO" in supported
        except Exception:
            return False

    def get_device_info(self, device_name: str) -> Dict[str, Any]:
        target = device_name.upper()
        matched = [d for d in self.available_devices if d.upper() == target or target.startswith(d.upper())]
        if not matched:
            return {"error": f"Device {device_name} not found"}

        dev = matched[0]
        info: Dict[str, Any] = {"device": dev}

        try:
            info["full_name"] = self._core.get_property(dev, "FULL_DEVICE_NAME")
        except Exception:
            pass

        try:
            info["architecture"] = self._core.get_property(dev, "DEVICE_ARCHITECTURE")
        except Exception:
            pass

        try:
            info["capabilities"] = list(self._core.get_property(dev, "OPTIMIZATION_CAPABILITIES"))
        except Exception:
            pass

        if "GPU" in dev:
            gpu_mem = self.get_gpu_memory_stats(dev)
            info["total_memory_gb"] = gpu_mem.get("total_memory_gb", 0.0)
            info["max_alloc_memory_gb"] = gpu_mem.get("max_alloc_memory_gb", 0.0)
            info["supports_offload_ratio"] = self.supports_offload_ratio(dev)
            try:
                info["execution_units"] = self._core.get_property(dev, "GPU_EXECUTION_UNITS_COUNT")
            except Exception:
                pass

        if "NPU" in dev:
            try:
                info["driver_version"] = str(self._core.get_property(dev, "NPU_DRIVER_VERSION"))
            except Exception:
                pass

        return info

    def get_all_devices_summary(self) -> List[Dict[str, Any]]:
        return [self.get_device_info(d) for d in self.available_devices]

    def inspect_model_dir(self, model_path: Union[str, Path]) -> ModelInspectResult:
        """分析本地模型資料夾或虛擬預估配置，識別是否為 35B-A3B MoE、是否含 MTP 與 Vision 子模型"""
        p = Path(model_path)
        res = ModelInspectResult(name=p.name, path=str(p))

        if not p.exists():
            # 若尚未下載，根據名稱進行啟發式估算 (例如 Qwen3.8-35B-A3B 或 Qwen3.6-35B-A3B)
            name_lower = p.name.lower()
            if "35b-a3b" in name_lower or "qwen3.6" in name_lower or "qwen3.8" in name_lower:
                res.architecture = "qwen3_5_moe"
                res.is_moe = True
                res.has_linear_attention = True
                res.has_mtp = True
                res.has_vision = True
                res.num_layers = 40
                res.num_experts = 256
                res.num_experts_per_tok = 8
                res.language_weight_gb = 16.75
                res.embedding_weight_gb = 0.47
                res.mtp_weight_gb = 0.41
                res.vision_weight_gb = 0.41
                res.total_weight_gb = 18.04
                res.estimated_dense_gb = 3.65
                res.estimated_routed_experts_gb = 13.57
            return res

        # 1. 檢查權重檔案大小
        lang_bin = p / "openvino_language_model.bin"
        mono_bin = p / "openvino_model.bin"
        emb_bin = p / "openvino_text_embeddings_model.bin"
        mtp_bin = p / "openvino_mtp_model.bin"
        vis_bin = p / "openvino_vision_embeddings_model.bin"
        vis_merger_bin = p / "openvino_vision_embeddings_merger_model.bin"

        if lang_bin.exists():
            res.format = "openvino_vlm_ir"
            res.language_weight_gb = round(lang_bin.stat().st_size / (1024 ** 3), 3)
        elif mono_bin.exists():
            res.format = "openvino_ir"
            res.language_weight_gb = round(mono_bin.stat().st_size / (1024 ** 3), 3)
        else:
            ggufs = list(p.glob("*.gguf"))
            if ggufs:
                res.format = "gguf"
                res.language_weight_gb = round(sum(g.stat().st_size for g in ggufs) / (1024 ** 3), 3)

        if emb_bin.exists():
            res.embedding_weight_gb = round(emb_bin.stat().st_size / (1024 ** 3), 3)
        if mtp_bin.exists() and (p / "openvino_mtp_model.xml").exists():
            res.has_mtp = True
            res.mtp_weight_gb = round(mtp_bin.stat().st_size / (1024 ** 3), 3)
        if vis_bin.exists() or vis_merger_bin.exists():
            res.has_vision = True
            v_bytes = (vis_bin.stat().st_size if vis_bin.exists() else 0) + (
                vis_merger_bin.stat().st_size if vis_merger_bin.exists() else 0
            )
            res.vision_weight_gb = round(v_bytes / (1024 ** 3), 3)

        res.total_weight_gb = round(
            res.language_weight_gb + res.embedding_weight_gb + res.mtp_weight_gb + res.vision_weight_gb,
            3,
        )

        # 2. 解析 config.json 與 openvino_config.json
        cfg_file = p / "config.json"
        if cfg_file.exists():
            try:
                cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
                # 支援 VLM 結構 (text_config 內部)
                text_cfg = cfg.get("text_config", cfg)
                model_type = cfg.get("model_type") or text_cfg.get("model_type", "unknown")
                res.architecture = model_type
                res.num_layers = text_cfg.get("num_hidden_layers", 0)
                res.num_experts = text_cfg.get("num_experts", text_cfg.get("n_routed_experts", 0))
                res.num_experts_per_tok = text_cfg.get(
                    "num_experts_per_tok", text_cfg.get("num_selected_experts", 0)
                )
                if res.num_experts > 1 or "moe" in model_type.lower() or "a3b" in p.name.lower():
                    res.is_moe = True
                if (
                    "linear" in str(text_cfg.get("layer_types", "")).lower()
                    or "deltanet" in str(text_cfg).lower()
                    or model_type in ("qwen3_5_moe", "qwen3_5", "qwen35moe")
                    or "35b-a3b" in p.name.lower()
                ):
                    res.has_linear_attention = True
            except Exception as e:
                logger.warning(f"Failed to parse config.json in {p}: {e}")

        ov_cfg_file = p / "openvino_config.json"
        if ov_cfg_file.exists():
            try:
                ov_cfg = json.loads(ov_cfg_file.read_text(encoding="utf-8"))
                q_cfg = ov_cfg.get("quantization_config", {})
                mode = q_cfg.get("mode") or ov_cfg.get("weight_format")
                if mode:
                    res.quantization = str(mode).upper()
            except Exception:
                pass

        # 3. 估算 Dense 與 Routed Experts 的權重占比
        if res.is_moe:
            # 以 Qwen3.6/3.8-35B-A3B 為例: 總參數量 35B，激活 3B (Shared + Dense 約占 18~22%，Routed Experts 約占 78~82%)
            active_ratio = 0.20
            if res.num_experts > 0 and res.num_experts_per_tok > 0:
                # Dense backbone (Attention/GDN + Shared Expert + Embeddings) 約占 18%~22%
                active_ratio = max(0.16, min(0.30, (res.num_experts_per_tok / res.num_experts) * 2.5 + 0.12))
            res.estimated_dense_gb = round(
                res.language_weight_gb * active_ratio + res.embedding_weight_gb, 2
            )
            res.estimated_routed_experts_gb = round(
                max(0.0, res.language_weight_gb * (1.0 - active_ratio)), 2
            )
        else:
            res.estimated_dense_gb = round(res.language_weight_gb + res.embedding_weight_gb, 2)
            res.estimated_routed_experts_gb = 0.0

        return res

    def plan_strata_execution(
        self,
        model_path: Union[str, Path],
        config: Optional[StrataConfig] = None,
    ) -> StrataPlan:
        """
        核心規劃器：結合 PolyStrata 顯存預算理念與 OpenVINO 2026 OFFLOAD_RATIO / KV 壓縮機制，
        計算出能在有限 VRAM (例如 6GB / 8GB / 12GB / 16GB iGPU) 上穩定運行 35B-A3B 的配置。
        """
        cfg = config or StrataConfig()
        model_info = self.inspect_model_dir(model_path)
        sys_mem = self.get_system_memory_info()
        target_dev = cfg.device.upper()

        gpu_stats = self.get_gpu_memory_stats("GPU")
        vram_total_gb = gpu_stats.get("max_alloc_memory_gb") or gpu_stats.get("total_memory_gb") or 8.0
        reserve_gb = round(cfg.vram_reserve_mib / 1024.0, 2)

        # 決定可用 VRAM 預算
        if cfg.vram_budget_gb is not None and cfg.vram_budget_gb > 0:
            vram_budget_gb = min(cfg.vram_budget_gb, max(2.0, vram_total_gb - 0.5))
        else:
            vram_budget_gb = round(max(2.0, vram_total_gb - reserve_gb), 2)

        notes: List[str] = []

        # 決定 KV Cache 預算 (GB)
        if cfg.kv_cache_size_gb is not None and cfg.kv_cache_size_gb > 0:
            kv_budget_gb = cfg.kv_cache_size_gb
        else:
            # 若是 Qwen3.6/3.8-35B-A3B (30 層 Gated DeltaNet + 10 層 Full Attention) + INT8 KV Cache，
            # 32K context 僅需約 0.6~1.2 GB
            if model_info.has_linear_attention:
                kv_budget_gb = 1.0 if cfg.kv_cache_precision == "u8" else 1.8
                notes.append(
                    "偵測到混合線性注意力架構 (Gated DeltaNet + Full Attention)，長上下文 KV 顯存消耗大幅降低。"
                )
            else:
                kv_budget_gb = 1.2 if cfg.kv_cache_precision == "u8" else 2.2
            # 對於小模型 (<2GB)，KV 預算無需過大
            if model_info.total_weight_gb > 0 and model_info.total_weight_gb < 2.0:
                kv_budget_gb = min(kv_budget_gb, 0.5)

        # 計算可用於模型權重的 GPU 顯存上限 (扣除 KV Cache 與約 0.6 GB 的運算暫存 Scratch)
        scratch_gb = 0.6 if model_info.total_weight_gb >= 4.0 else 0.2
        weight_vram_budget_gb = max(0.5, vram_budget_gb - kv_budget_gb - scratch_gb)

        # 決定 OFFLOAD_RATIO (0 ~ 100)
        if "GPU" not in target_dev:
            recommended_offload = 100
            notes.append(f"目標裝置為 {target_dev}，全部權重將於系統記憶體中執行。")
        elif cfg.offload_ratio is not None and cfg.offload_ratio >= 0:
            recommended_offload = max(0, min(100, int(cfg.offload_ratio)))
            notes.append(f"使用使用者顯式指定的專家卸載比例 OFFLOAD_RATIO={recommended_offload}%。")
        elif not model_info.is_moe and model_info.total_weight_gb <= weight_vram_budget_gb:
            recommended_offload = 0
            notes.append("模型大小可完全放入 GPU 顯存預算內 (OFFLOAD_RATIO=0%)。")
        elif model_info.is_moe:
            dense_gb = model_info.estimated_dense_gb
            experts_gb = max(0.1, model_info.estimated_routed_experts_gb)
            room_for_experts_gb = max(0.0, weight_vram_budget_gb - dense_gb)
            if room_for_experts_gb >= experts_gb:
                recommended_offload = 0
                notes.append("GPU 顯存充足，Dense 層與全部 100% 路由專家皆可常駐 VRAM。")
            else:
                # 計算可放入 VRAM 的熱點專家比例，並預留 5% 緩衝
                keep_ratio = max(0.0, min(1.0, (room_for_experts_gb / experts_gb) - 0.05))
                # 對齊至 5% 步進
                raw_offload = int(round((1.0 - keep_ratio) * 100 / 5.0) * 5)
                recommended_offload = max(10, min(100, raw_offload))
                notes.append(
                    f"PolyStrata 分層策略：將 Dense 主幹 ({dense_gb:.1f} GB) 與 {100 - recommended_offload}% 熱點專家駐留於 GPU VRAM，"
                    f"其餘 {recommended_offload}% 專家透過 OpenVINO OFFLOAD_RATIO 卸載至系統 RAM。"
                )
        else:
            # 大型 Dense 模型超出顯存預算時，同樣按比例設定 OFFLOAD_RATIO
            if model_info.total_weight_gb > weight_vram_budget_gb:
                keep_ratio = max(0.0, min(1.0, weight_vram_budget_gb / max(0.1, model_info.total_weight_gb)))
                recommended_offload = max(10, min(100, int(round((1.0 - keep_ratio) * 100 / 5.0) * 5)))
                notes.append(
                    f"模型權重 ({model_info.total_weight_gb:.1f} GB) 超出顯存預算 ({vram_budget_gb:.1f} GB)，"
                    f"自動啟用 OFFLOAD_RATIO={recommended_offload}%。"
                )
            else:
                recommended_offload = 0

        # 估算最終顯存與系統記憶體使用量
        if "GPU" in target_dev:
            if model_info.is_moe:
                experts_in_vram = model_info.estimated_routed_experts_gb * (1.0 - recommended_offload / 100.0)
                experts_in_ram = model_info.estimated_routed_experts_gb * (recommended_offload / 100.0)
                est_vram = round(model_info.estimated_dense_gb + experts_in_vram + kv_budget_gb + scratch_gb, 2)
                est_ram = round(experts_in_ram + (model_info.mtp_weight_gb if cfg.enable_mtp else 0.0) + 1.5, 2)
            else:
                w_vram = model_info.total_weight_gb * (1.0 - recommended_offload / 100.0)
                w_ram = model_info.total_weight_gb * (recommended_offload / 100.0)
                est_vram = round(w_vram + kv_budget_gb + scratch_gb, 2)
                est_ram = round(w_ram + 1.0, 2)
        else:
            est_vram = 0.0
            est_ram = round(model_info.total_weight_gb + kv_budget_gb + 1.5, 2)

        # MTP 與 Prefix Caching 協調規則 (OpenVINO GenAI 規範: 啟用 MTP 時需關閉 Prefix Caching)
        use_mtp = bool(cfg.enable_mtp and model_info.has_mtp)
        use_prefix_caching = bool(cfg.enable_prefix_caching and not use_mtp)
        if use_mtp and cfg.enable_prefix_caching:
            notes.append("已啟用內建 MTP 投機解碼草稿模型，自動停用 Prefix Caching 以符合 OpenVINO GenAI 管線要求。")

        # 組合 OpenVINO Plugin 底層屬性
        ov_plugin_config: Dict[str, Any] = {}
        if "GPU" in target_dev:
            ov_plugin_config["KV_CACHE_PRECISION"] = cfg.kv_cache_precision
            ov_plugin_config["GPU_ENABLE_LARGE_ALLOCATIONS"] = True
            if self.supports_offload_ratio("GPU") and recommended_offload > 0:
                ov_plugin_config["OFFLOAD_RATIO"] = recommended_offload

        return StrataPlan(
            model_info=model_info,
            target_device=target_dev,
            vram_total_gb=vram_total_gb,
            vram_budget_gb=vram_budget_gb,
            ram_total_gb=sys_mem["total_ram_gb"],
            ram_available_gb=sys_mem["available_ram_gb"],
            recommended_offload_ratio=recommended_offload,
            kv_cache_precision=cfg.kv_cache_precision,
            kv_cache_budget_gb=kv_budget_gb,
            estimated_vram_usage_gb=est_vram,
            estimated_ram_offload_gb=est_ram,
            hot_experts_in_vram_pct=100 - recommended_offload,
            cold_experts_in_ram_pct=recommended_offload,
            use_mtp=use_mtp,
            mtp_device=cfg.mtp_device.upper(),
            use_prefix_caching=use_prefix_caching,
            ov_plugin_config=ov_plugin_config,
            notes=notes,
        )


# 全域單例
device_manager = DeviceManager()
