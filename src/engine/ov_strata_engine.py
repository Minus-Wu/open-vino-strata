"""
OpenVINO-Strata 核心分層推理引擎 (OVStrataEngine)
結合 Intel OpenVINO 2026 GenAI 與 PolyStrata 顯存分層架構：
- 支援 MoE 專家權重分層卸載 (GPU OFFLOAD_RATIO + 系統 RAM 協同)
- 支援 KV Cache INT8 壓縮與 Gated DeltaNet 混合線性注意力
- 支援內建 MTP (Multi-Token Prediction) 投機解碼草稿模型
- 支援多模態圖片輸入 (Vision VLMPipeline / Image Tensor 解析)
- 支援 PolyStrata 風格之動態模型熱切換 (switch_model)、手動顯存釋放 (unload_model) 與閒置自動卸載 (Idle Watchdog)
"""

import base64
import gc
import io
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple, Union

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
import requests
from PIL import Image

from src.engine.device_manager import (
    device_manager,
    StrataConfig,
    StrataPlan,
    ModelInspectResult,
)
from src.engine.parser import format_tools_system_instruction

logger = logging.getLogger("ov_strata.engine")


class EngineMetrics:
    def __init__(self):
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.reasoning_tokens: int = 0
        self.total_tokens: int = 0
        self.load_time: float = 0.0
        self.ttft: float = 0.0            # Time to first token (seconds)
        self.total_time: float = 0.0      # Total generation time (seconds)
        self.prefill_tps: float = 0.0     # Prompt reading speed (tokens/s)
        self.tps: float = 0.0             # Decode speed (tokens/s)
        self.offload_ratio: int = 0       # MoE Expert offload ratio (%)
        self.vram_allocated_mb: float = 0.0
        self.ram_used_gb: float = 0.0
        self.mtp_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        decode_time = max(0.0001, self.total_time - self.ttft) if self.total_time > self.ttft else self.total_time
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "ttft_seconds": round(self.ttft, 4),
            "total_time_seconds": round(self.total_time, 4),
            "prefill_tokens_per_second": round(self.prefill_tps, 2),
            "tokens_per_second": round(self.tps, 2),
            "decode_tokens_per_second": round(self.completion_tokens / decode_time, 2) if self.completion_tokens > 0 else 0.0,
            "offload_ratio": self.offload_ratio,
            "vram_allocated_mb": round(self.vram_allocated_mb, 1),
            "ram_used_gb": round(self.ram_used_gb, 2),
            "mtp_enabled": self.mtp_enabled,
        }


def is_valid_ov_model_dir(p: Path) -> bool:
    """判斷資料夾是否包含可載入之 OpenVINO 模型 (單體 IR 或 VLM/MoE 分離式 IR)"""
    if not p.is_dir():
        return False
    return (
        (p / "openvino_model.xml").exists()
        or (p / "openvino_language_model.xml").exists()
    )


class OVStrataEngine:
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "GPU",
        strata_config: Optional[StrataConfig] = None,
        models_search_dirs: Optional[List[Path]] = None,
        auto_load: bool = True,
        **pipeline_kwargs,
    ):
        self.strata_config = strata_config or StrataConfig(device=device.upper())
        if device:
            self.strata_config.device = device.upper()
        if "offload_ratio" in pipeline_kwargs:
            self.strata_config.offload_ratio = pipeline_kwargs.pop("offload_ratio")
        if "vram_budget_gb" in pipeline_kwargs:
            self.strata_config.vram_budget_gb = pipeline_kwargs.pop("vram_budget_gb")

        self.pipeline_kwargs = pipeline_kwargs
        self.project_root = Path(__file__).resolve().parent.parent.parent

        # 支援同時搜尋本專案 models/ 與隔壁專案 ..\..\aio-mono\open-vino-llm\models
        default_dirs = [
            self.project_root / "models",
            self.project_root.parent.parent / "aio-mono" / "open-vino-llm" / "models",
        ]
        self.models_search_dirs: List[Path] = models_search_dirs or default_dirs

        self.model_path: Optional[Path] = Path(model_path) if model_path else None
        self.device: str = self.strata_config.device.upper()

        self._pipe: Optional[Union[ov_genai.LLMPipeline, ov_genai.VLMPipeline]] = None
        self._tokenizer: Optional[ov_genai.Tokenizer] = None
        self._model_name: str = self.model_path.name if self.model_path else "unloaded"
        self._is_vlm_pipeline: bool = False
        self._current_plan: Optional[StrataPlan] = None
        self._load_duration: float = 0.0
        self._last_active_time: float = time.time()

        # 單一模型推論執行緒池與非同步鎖，確保 C++ 管線與熱切換安全
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="OVStrataWorker")
        self._lock = asyncio.Lock()

        if auto_load and self.model_path is not None:
            self.load_model(self.model_path, self.device, self.strata_config)

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def current_device(self) -> str:
        return self.device

    @property
    def current_plan(self) -> Optional[StrataPlan]:
        return self._current_plan

    def discover_available_models(self) -> List[Dict[str, Any]]:
        """掃描所有模型目錄，回傳類似 PolyStrata `serve/models.py` 的模型註冊表清單"""
        seen_names = set()
        results: List[Dict[str, Any]] = []

        for base_dir in self.models_search_dirs:
            if not base_dir.exists():
                continue
            for sub in sorted(base_dir.iterdir()):
                if not sub.is_dir():
                    continue
                has_ov = is_valid_ov_model_dir(sub)
                has_gguf = any(sub.glob("*.gguf"))
                if not (has_ov or has_gguf):
                    continue
                if sub.name in seen_names:
                    continue
                seen_names.add(sub.name)

                plan = device_manager.plan_strata_execution(sub, self.strata_config)
                info = plan.model_info
                is_current = self.model_path is not None and sub.resolve() == self.model_path.resolve()
                status = "loaded" if (is_current and self.is_loaded) else ("unloaded" if is_current else "available")

                results.append({
                    "id": sub.name,
                    "path": str(sub),
                    "format": info.format,
                    "architecture": info.architecture,
                    "quantization": info.quantization,
                    "is_moe": info.is_moe,
                    "has_linear_attention": info.has_linear_attention,
                    "has_mtp": info.has_mtp,
                    "has_vision": info.has_vision,
                    "total_weight_gb": info.total_weight_gb,
                    "estimated_dense_gb": info.estimated_dense_gb,
                    "estimated_routed_experts_gb": info.estimated_routed_experts_gb,
                    "recommended_offload_ratio": plan.recommended_offload_ratio,
                    "estimated_vram_gb": plan.estimated_vram_usage_gb,
                    "estimated_ram_gb": plan.estimated_ram_offload_gb,
                    "current": is_current,
                    "status": status,
                })

        return results

    def resolve_model_path(self, model_name_or_path: Union[str, Path]) -> Path:
        """根據模型名稱或完整路徑解析真實模型目錄"""
        candidate = Path(model_name_or_path)
        if candidate.exists() and is_valid_ov_model_dir(candidate):
            return candidate.resolve()

        name_str = str(model_name_or_path).strip()
        # 1. 精確匹配
        for base_dir in self.models_search_dirs:
            if not base_dir.exists():
                continue
            exact = base_dir / name_str
            if exact.exists() and is_valid_ov_model_dir(exact):
                return exact.resolve()

        # 2. 模糊子字串匹配 (如同 PolyStrata 支援 `--model qwen3.6` 簡寫)
        name_lower = name_str.lower()
        for base_dir in self.models_search_dirs:
            if not base_dir.exists():
                continue
            for sub in sorted(base_dir.iterdir()):
                if is_valid_ov_model_dir(sub) and name_lower in sub.name.lower():
                    return sub.resolve()

        raise FileNotFoundError(f"找不到符合 '{model_name_or_path}' 的 OpenVINO 模型資料夾。")

    def _build_ov_properties(self, plan: StrataPlan, model_path: Path) -> Dict[str, Any]:
        """建立傳遞給 LLMPipeline / VLMPipeline 的 OpenVINO 底層配置"""
        props: Dict[str, Any] = dict(plan.ov_plugin_config)
        props.update(self.pipeline_kwargs)

        # OpenVINO 2026 Intel GPU MoE OTD (On-The-fly Offloading) 要求顯式提供 WEIGHTS_PATH 指向 .bin 權重檔
        if props.get("OFFLOAD_RATIO", 0) > 0 and "WEIGHTS_PATH" not in props:
            for bin_name in ("openvino_language_model.bin", "openvino_model.bin"):
                bin_candidate = model_path / bin_name
                if bin_candidate.exists():
                    props["WEIGHTS_PATH"] = str(bin_candidate.resolve())
                    break

        # 設定編譯快取目錄 (.ov_cache)；當啟用 MoE OTD (OFFLOAD_RATIO > 0) 時直接由 .bin 串流權重，不啟用 CACHE_DIR 以免衝突
        if self.strata_config.enable_compile_cache and props.get("OFFLOAD_RATIO", 0) == 0:
            cache_dir = self.project_root / ".ov_cache" / f"{model_path.name}_{plan.target_device}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            props["CACHE_DIR"] = str(cache_dir)

        # 若啟用內建 MTP 草稿模型
        if plan.use_mtp and (model_path / "openvino_mtp_model.xml").exists():
            try:
                draft = ov_genai.draft_model(str(model_path), plan.mtp_device)
                props["draft_model"] = draft
                sched_cfg = ov_genai.SchedulerConfig()
                sched_cfg.enable_prefix_caching = False
                if plan.kv_cache_budget_gb > 0:
                    sched_cfg.cache_size = plan.kv_cache_budget_gb
                props["scheduler_config"] = sched_cfg
                logger.info(
                    f"Enabled built-in MTP speculative decoding draft model on [{plan.mtp_device}]"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize MTP draft model ({e}), continuing without MTP.")

        return props

    def load_model(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        strata_config: Optional[StrataConfig] = None,
    ) -> StrataPlan:
        """載入並編譯模型至指定硬體裝置，自動套用 Strata OFFLOAD_RATIO 與顯存優化參數"""
        if strata_config is not None:
            self.strata_config = strata_config
        if device is not None:
            self.strata_config.device = device.upper()
            self.device = device.upper()

        target_path = self.resolve_model_path(model_path or self.model_path)
        target_device = self.strata_config.device.upper()

        if not device_manager.is_device_available(target_device):
            raise ValueError(f"Target device [{target_device}] is not available on this system.")

        # 若目前已有載入模型，先徹底釋放舊模型顯存 (PolyStrata 規範：兩模型絕不重疊占據顯存)
        if self._pipe is not None:
            self._release_pipeline_resources()

        plan = device_manager.plan_strata_execution(target_path, self.strata_config)
        props = self._build_ov_properties(plan, target_path)

        logger.info(
            f"Loading model [{target_path.name}] on [{target_device}] "
            f"(OFFLOAD_RATIO={plan.recommended_offload_ratio}%, KV={plan.kv_cache_precision}, "
            f"Est. VRAM={plan.estimated_vram_usage_gb}GB, Est. RAM={plan.estimated_ram_offload_gb}GB)..."
        )
        t0 = time.perf_counter()

        self._tokenizer = ov_genai.Tokenizer(str(target_path))
        has_vlm_files = (target_path / "openvino_vision_embeddings_model.xml").exists() and (
            target_path / "openvino_language_model.xml"
        ).exists()
        use_vlm_pipe = has_vlm_files and (
            self.strata_config.enable_vision or not (target_path / "openvino_model.xml").exists()
        )

        try:
            if use_vlm_pipe:
                self._pipe = ov_genai.VLMPipeline(str(target_path), target_device, **props)
                self._is_vlm_pipeline = True
            else:
                self._pipe = ov_genai.LLMPipeline(
                    str(target_path), self._tokenizer, target_device, **props
                )
                self._is_vlm_pipeline = False
        except Exception as first_err:
            logger.warning(
                f"Primary pipeline init with extended Strata properties raised ({first_err}); "
                "retrying with essential properties..."
            )
            # 降級容錯：保留關鍵 OFFLOAD_RATIO、WEIGHTS_PATH 與 KV_CACHE_PRECISION 屬性後重試
            safe_props = {
                k: v
                for k, v in props.items()
                if k in ("OFFLOAD_RATIO", "WEIGHTS_PATH", "KV_CACHE_PRECISION", "CACHE_DIR")
            }
            try:
                if use_vlm_pipe:
                    self._pipe = ov_genai.VLMPipeline(str(target_path), target_device, **safe_props)
                    self._is_vlm_pipeline = True
                else:
                    self._pipe = ov_genai.LLMPipeline(
                        str(target_path), self._tokenizer, target_device, **safe_props
                    )
                    self._is_vlm_pipeline = False
            except Exception as second_err:
                logger.warning(f"Retry with safe_props raised ({second_err}); falling back to base init...")
                if use_vlm_pipe:
                    self._pipe = ov_genai.VLMPipeline(str(target_path), target_device)
                    self._is_vlm_pipeline = True
                else:
                    self._pipe = ov_genai.LLMPipeline(str(target_path), self._tokenizer, target_device)
                    self._is_vlm_pipeline = False

        self.model_path = target_path
        self._model_name = target_path.name
        self.device = target_device
        self._current_plan = plan
        self._load_duration = time.perf_counter() - t0
        self._last_active_time = time.time()

        logger.info(
            f"Model [{self._model_name}] compiled on [{self.device}] in {self._load_duration:.2f}s "
            f"(Offload={plan.recommended_offload_ratio}%, VLM={self._is_vlm_pipeline})"
        )
        return plan

    def _release_pipeline_resources(self):
        """內部同步釋放 C++ Pipeline 與 Tokenizer 並觸發 GC 回收顯存"""
        old_name = self._model_name
        self._pipe = None
        self._tokenizer = None
        self._is_vlm_pipeline = False
        gc.collect()
        logger.info(f"Released VRAM and host memory for model [{old_name}].")

    async def unload_model(self) -> Dict[str, Any]:
        """非同步釋放目前載入的模型，將顯存與記憶體歸還作業系統 (對標 PolyStrata /api/models/unload)"""
        async with self._lock:
            before_gpu = device_manager.get_gpu_memory_stats("GPU")
            before_ram = device_manager.get_system_memory_info()
            unloaded_name = self._model_name
            self._release_pipeline_resources()
            after_gpu = device_manager.get_gpu_memory_stats("GPU")
            after_ram = device_manager.get_system_memory_info()
            return {
                "status": "unloaded",
                "unloaded_model": unloaded_name,
                "vram_before_mb": before_gpu.get("allocated_memory_mb", 0.0),
                "vram_after_mb": after_gpu.get("allocated_memory_mb", 0.0),
                "ram_available_before_gb": before_ram.get("available_ram_gb", 0.0),
                "ram_available_after_gb": after_ram.get("available_ram_gb", 0.0),
            }

    async def switch_model(
        self,
        model_name_or_path: Union[str, Path],
        device: Optional[str] = None,
        strata_config: Optional[StrataConfig] = None,
    ) -> Dict[str, Any]:
        """
        PolyStrata 風格之動態模型熱切換 (Service.switch_model)：
        1. 等待執行中的請求完成
        2. 先卸載舊模型釋放顯存
        3. 載入新模型；若失敗則自動嘗試恢復舊模型
        """
        async with self._lock:
            prev_path = self.model_path
            prev_device = self.device
            prev_config = self.strata_config

            target_path = self.resolve_model_path(model_name_or_path)
            self._release_pipeline_resources()

            loop = asyncio.get_running_loop()
            try:
                plan = await loop.run_in_executor(
                    self._executor,
                    lambda: self.load_model(target_path, device or self.device, strata_config or self.strata_config),
                )
                return {
                    "status": "loaded",
                    "model": self._model_name,
                    "device": self.device,
                    "load_time_seconds": round(self._load_duration, 2),
                    "strata_plan": plan.to_dict(),
                }
            except Exception as e:
                logger.error(f"Failed to switch to model [{target_path}]: {e}")
                if prev_path is not None and prev_path.exists():
                    logger.info(f"Rolling back to previous model [{prev_path.name}]...")
                    try:
                        await loop.run_in_executor(
                            self._executor,
                            lambda: self.load_model(prev_path, prev_device, prev_config),
                        )
                    except Exception as rollback_err:
                        logger.error(f"Rollback also failed: {rollback_err}")
                raise e

    async def ensure_loaded(self):
        """若模型處於 Unloaded 狀態 (例如剛啟動或被閒置自動卸載)，於收到請求時自動喚醒載入"""
        if self._pipe is not None:
            self._last_active_time = time.time()
            return
        if self.model_path is None:
            models = self.discover_available_models()
            if not models:
                raise RuntimeError("No available OpenVINO models found in models/ directory.")
            self.model_path = Path(models[0]["path"])

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            lambda: self.load_model(self.model_path, self.device, self.strata_config),
        )

    def count_tokens(self, text: str) -> int:
        """計算給定文字的 token 數量"""
        if not self._tokenizer or not text:
            return max(1, len(text) // 4) if text else 0
        try:
            tokens = self._tokenizer.encode(text)
            return int(tokens.input_ids.shape[-1])
        except Exception:
            return len(text.split())

    def load_image_tensor(self, image_source: str) -> ov.Tensor:
        """
        將 Base64 Data URI (`data:image/...;base64,...`)、HTTP URL 或本機檔案路徑
        解析為 OpenVINO VLMPipeline 所需的 `ov.Tensor` (NHWC uint8)。
        """
        if image_source.startswith("data:image"):
            _, b64_data = image_source.split(",", 1)
            raw_bytes = base64.b64decode(b64_data)
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        elif image_source.startswith(("http://", "https://")):
            resp = requests.get(image_source, timeout=15)
            resp.raise_for_status()
            pil_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        else:
            pil_img = Image.open(image_source).convert("RGB")

        arr = np.array(pil_img, dtype=np.uint8)[None]
        return ov.Tensor(arr)

    def extract_text_and_images(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, str]], List[ov.Tensor]]:
        """
        從 OpenAI / Anthropic 多模態訊息列表中分離純文字訊息與圖片 Tensor 列表。
        """
        normalized_messages: List[Dict[str, str]] = []
        image_tensors: List[ov.Tensor] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                normalized_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                text_parts: List[str] = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict):
                        p_type = part.get("type", "")
                        if p_type == "text":
                            text_parts.append(str(part.get("text", "")))
                        elif p_type == "image_url":
                            img_url_obj = part.get("image_url", {})
                            url = img_url_obj.get("url", "") if isinstance(img_url_obj, dict) else str(img_url_obj)
                            if url:
                                try:
                                    image_tensors.append(self.load_image_tensor(url))
                                except Exception as e:
                                    logger.warning(f"Failed to load image from {url[:60]}: {e}")
                        elif p_type == "image":
                            # Anthropic image block: {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
                            src = part.get("source", {})
                            if src.get("type") == "base64" and src.get("data"):
                                media = src.get("media_type", "image/png")
                                data_uri = f"data:{media};base64,{src['data']}"
                                try:
                                    image_tensors.append(self.load_image_tensor(data_uri))
                                except Exception as e:
                                    logger.warning(f"Failed to decode Anthropic base64 image: {e}")
                            elif src.get("type") == "url" and src.get("url"):
                                try:
                                    image_tensors.append(self.load_image_tensor(src["url"]))
                                except Exception as e:
                                    logger.warning(f"Failed to load Anthropic URL image: {e}")
                normalized_messages.append({"role": role, "content": "\n".join(text_parts)})
            else:
                normalized_messages.append({"role": role, "content": str(content)})

        return normalized_messages, image_tensors

    def format_chat_prompt(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
        add_generation_prompt: bool = True,
    ) -> Tuple[str, bool, List[ov.Tensor]]:
        """
        套用模型的 Jinja2 Chat Template 將多輪對話與工具定義轉換為推論 Prompt。
        回傳: (formatted_prompt, prompt_opened_think, image_tensors)
        """
        text_messages, image_tensors = self.extract_text_and_images(messages)

        # 若提供 tools，將工具規範附加至 system 訊息中
        if tools:
            tool_sys_text = format_tools_system_instruction(tools)
            if text_messages and text_messages[0].get("role") == "system":
                text_messages[0] = {
                    "role": "system",
                    "content": text_messages[0]["content"] + tool_sys_text,
                }
            else:
                text_messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant." + tool_sys_text,
                    },
                )

        prompt_str = ""
        if self._tokenizer:
            try:
                prompt_str = self._tokenizer.apply_chat_template(
                    text_messages, add_generation_prompt=add_generation_prompt
                )
            except Exception as e:
                logger.warning(f"Failed to apply tokenizer chat template ({e}), using ChatML fallback.")

        if not prompt_str:
            prompt_parts = []
            for msg in text_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            if add_generation_prompt:
                prompt_parts.append("<|im_start|>assistant\n")
            prompt_str = "".join(prompt_parts)

        # 處理 Qwen3 / Qwen3.6 / Qwen3.8 的 enable_thinking 控制
        # 如同 PolyStrata INTEL.md 所述：若設定 enable_thinking=False，在 assistant 開頭預填空的 <think>\n\n</think>\n\n
        # 可讓推理模型直接跳過冗長思考、立即給出回答，節省寶貴的 token 預算。
        prompt_opened_think = False
        if enable_thinking is False:
            if prompt_str.rstrip().endswith("<think>"):
                prompt_str = prompt_str.rstrip() + "\n\n</think>\n\n"
            elif not prompt_str.rstrip().endswith("</think>"):
                prompt_str = prompt_str + "<think>\n\n</think>\n\n"
        elif enable_thinking is True:
            if prompt_str.rstrip().endswith("<think>"):
                prompt_opened_think = True
            elif "35b-a3b" in self._model_name.lower() or "deepseek-r1" in self._model_name.lower():
                prompt_str = prompt_str + "<think>\n"
                prompt_opened_think = True
        else:
            if prompt_str.rstrip().endswith("<think>"):
                prompt_opened_think = True

        return prompt_str, prompt_opened_think, image_tensors

    def _build_generation_config(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_strings: Optional[List[str]] = None,
        do_sample: bool = True,
        apply_chat_template: bool = False,
    ) -> ov_genai.GenerationConfig:
        """建立符合裝置與 MTP 限制的 GenerationConfig"""
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = max_new_tokens
        config.apply_chat_template = apply_chat_template

        use_mtp = bool(self._current_plan and self._current_plan.use_mtp)

        # 若在 NPU 上推論，或啟用了 OpenVINO GenAI MTP 草稿模型，限定 Greedy Decoding
        if "NPU" in self.device or use_mtp:
            config.do_sample = False
            if use_mtp:
                config.num_assistant_tokens = max(1, self.strata_config.num_assistant_tokens)
        else:
            config.do_sample = do_sample
            if do_sample:
                config.temperature = max(0.01, temperature)
                config.top_p = min(1.0, max(0.01, top_p))
                config.top_k = max(1, top_k)

        if repetition_penalty != 1.0:
            config.repetition_penalty = repetition_penalty

        if stop_strings:
            config.stop_strings = set(stop_strings)

        return config

    def _populate_strata_metrics(self, metrics: EngineMetrics):
        """填入 Strata 顯存與卸載指標"""
        if self._current_plan:
            metrics.offload_ratio = self._current_plan.recommended_offload_ratio
            metrics.mtp_enabled = self._current_plan.use_mtp
        gpu_stats = device_manager.get_gpu_memory_stats("GPU")
        metrics.vram_allocated_mb = gpu_stats.get(
            "allocated_memory_mb",
            (self._current_plan.estimated_vram_usage_gb * 1024.0) if self._current_plan else 0.0,
        )
        sys_mem = device_manager.get_system_memory_info()
        metrics.ram_used_gb = sys_mem.get("used_ram_gb", 0.0)

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_strings: Optional[List[str]] = None,
        do_sample: bool = True,
        images: Optional[List[ov.Tensor]] = None,
    ) -> Tuple[str, EngineMetrics]:
        """非同步非串流生成，回傳完整文字與 Strata 效能指標"""
        await self.ensure_loaded()

        # 若已由 format_chat_prompt 套用 ChatML (<|im_start|>) 且無圖片，停用底層重複套用 chat_template
        needs_vlm_template = bool(self._is_vlm_pipeline and images)
        gen_config = self._build_generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_strings=stop_strings,
            do_sample=do_sample,
            apply_chat_template=needs_vlm_template,
        )

        metrics = EngineMetrics()
        metrics.prompt_tokens = self.count_tokens(prompt)
        metrics.load_time = self._load_duration

        first_token_time: Optional[float] = None
        generated_pieces: List[str] = []

        def _streamer(token_text: str) -> bool:
            nonlocal first_token_time
            if first_token_time is None:
                first_token_time = time.perf_counter()
            generated_pieces.append(token_text)
            return False

        async with self._lock:
            self._last_active_time = time.time()
            loop = asyncio.get_running_loop()
            t_start = time.perf_counter()

            def _sync_generate():
                if self._is_vlm_pipeline:
                    if images:
                        return self._pipe.generate(
                            prompt, images=images, generation_config=gen_config, streamer=_streamer
                        )
                    return self._pipe.generate(
                        prompt, generation_config=gen_config, streamer=_streamer
                    )
                return self._pipe.generate(prompt, gen_config, _streamer)

            res = await loop.run_in_executor(self._executor, _sync_generate)
            t_end = time.perf_counter()
            self._last_active_time = time.time()

        output_text = "".join(generated_pieces) if generated_pieces else (
            res.texts[0] if hasattr(res, "texts") and res.texts else str(res)
        )

        metrics.total_time = t_end - t_start
        metrics.ttft = (first_token_time - t_start) if first_token_time else metrics.total_time
        metrics.completion_tokens = self.count_tokens(output_text)
        metrics.total_tokens = metrics.prompt_tokens + metrics.completion_tokens
        metrics.prefill_tps = (
            metrics.prompt_tokens / metrics.ttft if metrics.ttft > 0 else 0.0
        )
        metrics.tps = (
            metrics.completion_tokens / metrics.total_time if metrics.total_time > 0 else 0.0
        )
        self._populate_strata_metrics(metrics)

        return output_text, metrics

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_strings: Optional[List[str]] = None,
        do_sample: bool = True,
        images: Optional[List[ov.Tensor]] = None,
    ) -> AsyncGenerator[Tuple[str, Optional[EngineMetrics]], None]:
        """非同步串流生成產生器 (SSE-ready)，每產出 token 即 yield 出去，最後一個 chunk 附帶完整 Strata 指標"""
        await self.ensure_loaded()

        needs_vlm_template = bool(self._is_vlm_pipeline and images)
        gen_config = self._build_generation_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_strings=stop_strings,
            do_sample=do_sample,
            apply_chat_template=needs_vlm_template,
        )

        metrics = EngineMetrics()
        metrics.prompt_tokens = self.count_tokens(prompt)
        metrics.load_time = self._load_duration

        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        first_token_time: Optional[float] = None
        generated_pieces: List[str] = []

        def _streamer(token_text: str) -> bool:
            nonlocal first_token_time
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            loop.call_soon_threadsafe(queue.put_nowait, token_text)
            return False

        def _run_sync_pipeline():
            try:
                if self._is_vlm_pipeline:
                    if images:
                        self._pipe.generate(
                            prompt, images=images, generation_config=gen_config, streamer=_streamer
                        )
                    else:
                        self._pipe.generate(
                            prompt, generation_config=gen_config, streamer=_streamer
                        )
                else:
                    self._pipe.generate(prompt, gen_config, _streamer)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        async with self._lock:
            self._last_active_time = time.time()
            t_start = time.perf_counter()
            future = loop.run_in_executor(self._executor, _run_sync_pipeline)

            while True:
                token = await queue.get()
                if token is None:
                    break
                generated_pieces.append(token)
                yield token, None

            await future
            t_end = time.perf_counter()
            self._last_active_time = time.time()

        metrics.total_time = t_end - t_start
        metrics.ttft = (first_token_time - t_start) if first_token_time else metrics.total_time
        full_text = "".join(generated_pieces)
        metrics.completion_tokens = self.count_tokens(full_text)
        metrics.total_tokens = metrics.prompt_tokens + metrics.completion_tokens
        metrics.prefill_tps = (
            metrics.prompt_tokens / metrics.ttft if metrics.ttft > 0 else 0.0
        )
        metrics.tps = (
            metrics.completion_tokens / metrics.total_time if metrics.total_time > 0 else 0.0
        )
        self._populate_strata_metrics(metrics)

        yield "", metrics

    def get_info(self) -> Dict[str, Any]:
        """取得引擎、Strata 分層配置與硬體運作資訊"""
        gpu_stats = device_manager.get_gpu_memory_stats("GPU")
        sys_mem = device_manager.get_system_memory_info()
        return {
            "model_name": self._model_name,
            "model_path": str(self.model_path) if self.model_path else None,
            "loaded": self.is_loaded,
            "is_vlm": self._is_vlm_pipeline,
            "device": self.device,
            "load_time_seconds": round(self._load_duration, 2),
            "strata_config": self.strata_config.to_dict(),
            "strata_plan": self._current_plan.to_dict() if self._current_plan else None,
            "device_info": device_manager.get_device_info(self.device),
            "gpu_memory": gpu_stats,
            "system_memory": sys_mem,
        }
