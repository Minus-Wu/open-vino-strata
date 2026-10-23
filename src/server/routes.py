"""
OpenAI / Anthropic / PolyStrata 相容路由定義：
- GET  /health
- GET  /v1/models
- POST /v1/chat/completions (支援 SSE 串流、<think> reasoning_content、<tool_call> 與 Vision 圖片輸入)
- POST /v1/completions
- POST /v1/messages (Anthropic Messages API 相容端點，支援 Claude Code 等客戶端)
- GET  /api/models, /api/models/current (PolyStrata 模型清單與狀態查詢)
- POST /api/models/load, /api/models/unload (PolyStrata 動態模型熱切換與顯存釋放)
- POST /api/strata/plan (Strata 顯存預算與專家卸載比例預覽試算)
"""

import json
import logging
import uuid
from typing import AsyncGenerator, Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from src.server.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    DeltaMessage,
    UsageInfo,
    ModelList,
    ModelCard,
    CompletionRequest,
    AnthropicMessageRequest,
    ModelLoadRequest,
    StrataPlanRequest,
)
from src.engine.device_manager import device_manager, StrataConfig
from src.engine.ov_strata_engine import OVStrataEngine, EngineMetrics
from src.engine.parser import parse_full_output, IncrementalStreamParser

logger = logging.getLogger("ov_strata.routes")
router = APIRouter()

_engine: Optional[OVStrataEngine] = None
_sub_engine: Optional[OVStrataEngine] = None


def get_engine() -> OVStrataEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Inference engine is not initialized yet.")
    return _engine


def set_engine(engine: OVStrataEngine):
    global _engine
    _engine = engine


def get_sub_engine() -> Optional[OVStrataEngine]:
    return _sub_engine


def set_sub_engine(sub_engine: Optional[OVStrataEngine]):
    global _sub_engine
    _sub_engine = sub_engine


def resolve_engine_for_model(requested_model: Optional[str] = None) -> tuple[OVStrataEngine, bool]:
    """
    根據請求的模型名稱決定分流目標引擎。
    回傳: (選定引擎, 是否為副引擎)

    分流規則：
    1. 若未載入副引擎，一律使用主引擎。
    2. 若請求模型符合下列特徵，優先路由至副引擎 (NPU / CPU 副模型)：
       - 包含 'haiku' (Claude Code / Anthropic fast 模型)
       - 包含 'small', 'fast', 'mini', 'flash'
       - 精確匹配副引擎的 model_name 或路徑名稱
    3. 若明確要求 35b / sonnet / opus 或主模型名稱，一律走主引擎。
    """
    primary = get_engine()
    if _sub_engine is None or not _sub_engine.is_loaded:
        return primary, False

    if not requested_model:
        return primary, False

    m_lower = requested_model.strip().lower()

    # 明確指名主模型名稱或 35b / sonnet / opus 時強制主引擎
    if (
        "35b" in m_lower
        or "sonnet" in m_lower
        or "opus" in m_lower
        or m_lower == primary.model_name.lower()
    ):
        return primary, False

    is_fast_model = (
        "haiku" in m_lower
        or "small" in m_lower
        or "fast" in m_lower
        or "mini" in m_lower
        or "flash" in m_lower
        or m_lower == _sub_engine.model_name.lower()
        or (_sub_engine.model_path and m_lower == _sub_engine.model_path.name.lower())
    )

    if is_fast_model:
        return _sub_engine, True

    return primary, False


def _resolve_thinking_flag(request: ChatCompletionRequest) -> Optional[bool]:
    """解析請求中的思考模式設定 (支援 enable_thinking、chat_template_kwargs 與 reasoning_effort)"""
    if request.enable_thinking is not None:
        return request.enable_thinking
    if request.chat_template_kwargs and "enable_thinking" in request.chat_template_kwargs:
        return bool(request.chat_template_kwargs["enable_thinking"])
    if request.reasoning_effort is not None:
        effort = request.reasoning_effort.strip().lower()
        if effort in ("none", "off", "false", "disabled"):
            return False
        if effort in ("low", "medium", "high", "true", "on"):
            return True
    return None


def _build_usage_info(metrics: EngineMetrics, reasoning_tokens: int = 0) -> UsageInfo:
    m_dict = metrics.to_dict()
    return UsageInfo(
        prompt_tokens=metrics.prompt_tokens,
        completion_tokens=metrics.completion_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=metrics.total_tokens,
        ttft_seconds=m_dict["ttft_seconds"],
        prefill_tokens_per_second=m_dict["prefill_tokens_per_second"],
        tokens_per_second=m_dict["tokens_per_second"],
        decode_tokens_per_second=m_dict["decode_tokens_per_second"],
        offload_ratio=m_dict["offload_ratio"],
        vram_allocated_mb=m_dict["vram_allocated_mb"],
        ram_used_gb=m_dict["ram_used_gb"],
        mtp_enabled=m_dict["mtp_enabled"],
    )


@router.get("/health")
async def health_check(engine: OVStrataEngine = Depends(get_engine)):
    info = engine.get_info()
    plan = info.get("strata_plan") or {}
    resp = {
        "status": "healthy",
        "model": engine.model_name,
        "loaded": engine.is_loaded,
        "device": engine.current_device,
        "offload_ratio": plan.get("recommended_offload_ratio", 0),
        "estimated_vram_gb": plan.get("estimated_vram_usage_gb", 0.0),
        "estimated_ram_gb": plan.get("estimated_ram_offload_gb", 0.0),
        "gpu_memory": info.get("gpu_memory", {}),
        "system_memory": info.get("system_memory", {}),
    }
    if _sub_engine is not None and _sub_engine.is_loaded:
        resp["sub_engine"] = {
            "model": _sub_engine.model_name,
            "device": _sub_engine.current_device,
            "loaded": _sub_engine.is_loaded,
        }
    return resp


@router.get("/v1/models", response_model=ModelList)
async def list_models(engine: OVStrataEngine = Depends(get_engine)):
    discovered = engine.discover_available_models()
    dev_info = device_manager.get_device_info(engine.current_device)
    cards: List[ModelCard] = []

    for m in discovered:
        is_sub = bool(_sub_engine and _sub_engine.is_loaded and m["id"] == _sub_engine.model_name)
        active_dev = _sub_engine.current_device if is_sub else (engine.current_device if m["current"] else None)
        cards.append(
            ModelCard(
                id=m["id"],
                device=active_dev,
                loaded=m["status"] == "loaded" or is_sub,
                offload_ratio=0 if is_sub else m["recommended_offload_ratio"],
                estimated_vram_gb=0.0 if is_sub else m["estimated_vram_gb"],
                device_info=device_manager.get_device_info(active_dev) if active_dev else None,
            )
        )

    # 虛擬 Claude Code / Anthropic 相容別名，方便客戶端免設定映射
    if _sub_engine and _sub_engine.is_loaded:
        sub_dev_info = device_manager.get_device_info(_sub_engine.current_device)
        cards.append(
            ModelCard(
                id="claude-3-5-haiku-20241022",
                device=_sub_engine.current_device,
                loaded=True,
                offload_ratio=0,
                estimated_vram_gb=0.0,
                device_info=sub_dev_info,
            )
        )
    if engine.model_name:
        plan = engine.current_plan
        cards.append(
            ModelCard(
                id="claude-3-7-sonnet-20250219",
                device=engine.current_device,
                loaded=engine.is_loaded,
                offload_ratio=plan.recommended_offload_ratio if plan else 0,
                estimated_vram_gb=plan.estimated_vram_usage_gb if plan else 0.0,
                device_info=dev_info,
            )
        )

    return ModelList(data=cards)


# =====================================================================
# PolyStrata 風格動態模型與顯存分層管理 API (/api/models/* & /api/strata/*)
# =====================================================================

@router.get("/api/server/logs")
async def api_get_server_logs(tail: int = 300):
    """讀取伺服器最新日誌 (供 Chat 介面或監控工具檢視)"""
    project_root = Path(__file__).resolve().parent.parent.parent
    log_path = project_root / "logs" / "server.log"
    if not log_path.exists():
        return {
            "lines": ["(日誌檔案尚未產生或伺服器正在啟動中)"],
            "total_lines": 0,
            "file_size_bytes": 0,
        }
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            tail_lines = all_lines[-tail:] if len(all_lines) > tail else all_lines
            return {
                "lines": [line.rstrip("\r\n") for line in tail_lines],
                "total_lines": len(all_lines),
                "file_size_bytes": log_path.stat().st_size,
            }
    except Exception as e:
        return {"lines": [f"讀取日誌失敗: {e}"], "total_lines": 0, "file_size_bytes": 0}


@router.get("/api/models")
async def api_list_models(engine: OVStrataEngine = Depends(get_engine)):
    res = {
        "current": engine.model_name if engine.is_loaded else None,
        "loaded": engine.is_loaded,
        "device": engine.current_device,
        "models": engine.discover_available_models(),
    }
    if _sub_engine is not None and _sub_engine.is_loaded:
        res["sub_engine"] = {
            "model": _sub_engine.model_name,
            "device": _sub_engine.current_device,
            "loaded": _sub_engine.is_loaded,
        }
    return res


@router.get("/api/models/current")
async def api_current_model(engine: OVStrataEngine = Depends(get_engine)):
    return engine.get_info()


@router.post("/api/models/load")
async def api_load_model(
    req: ModelLoadRequest,
    engine: OVStrataEngine = Depends(get_engine),
):
    new_cfg = StrataConfig(
        device=(req.device or engine.current_device).upper(),
        offload_ratio=req.offload_ratio if req.offload_ratio is not None else engine.strata_config.offload_ratio,
        vram_budget_gb=req.vram_budget_gb if req.vram_budget_gb is not None else engine.strata_config.vram_budget_gb,
        kv_cache_precision=req.kv_cache_precision or engine.strata_config.kv_cache_precision,
        enable_mtp=req.enable_mtp if req.enable_mtp is not None else engine.strata_config.enable_mtp,
        mtp_device=req.mtp_device or engine.strata_config.mtp_device,
        enable_vision=req.enable_vision if req.enable_vision is not None else engine.strata_config.enable_vision,
    )
    try:
        res = await engine.switch_model(req.model, device=new_cfg.device, strata_config=new_cfg)
        return res
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@router.post("/api/models/unload")
async def api_unload_model(engine: OVStrataEngine = Depends(get_engine)):
    return await engine.unload_model()


@router.post("/api/strata/plan")
async def api_strata_plan(req: StrataPlanRequest):
    cfg = StrataConfig(
        device=req.device.upper(),
        offload_ratio=req.offload_ratio,
        vram_budget_gb=req.vram_budget_gb,
        vram_reserve_mib=req.vram_reserve_mib,
        kv_cache_precision=req.kv_cache_precision,
        enable_mtp=req.enable_mtp,
        mtp_device=req.mtp_device,
    )
    plan = device_manager.plan_strata_execution(req.model, cfg)
    return plan.to_dict()


# =====================================================================
# OpenAI 相容端點 (/v1/chat/completions & /v1/completions)
# =====================================================================

@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    engine: OVStrataEngine = Depends(get_engine),
):
    target_engine, is_sub = resolve_engine_for_model(request.model)

    # 若非副引擎且請求指定了特定本地模型，支援主引擎自動熱切換
    if (
        not is_sub
        and request.model
        and request.model not in ("default", "auto", target_engine.model_name)
    ):
        try:
            target_path = target_engine.resolve_model_path(request.model)
            if target_engine.model_path is None or target_path != target_engine.model_path.resolve():
                await target_engine.switch_model(target_path)
        except FileNotFoundError:
            # 若客戶端傳入外部別名，直接使用目前模型回應
            pass

    if is_sub:
        logger.info(
            f"Routing chat completion [model='{request.model}'] -> SUB-ENGINE "
            f"[{target_engine.current_device}: {target_engine.model_name}]"
        )
    else:
        logger.info(
            f"Routing chat completion [model='{request.model}'] -> PRIMARY "
            f"[{target_engine.current_device}: {target_engine.model_name}]"
        )

    req_id = f"chatcmpl-{uuid.uuid4().hex}"
    raw_messages = [msg.model_dump() for msg in request.messages]
    enable_thinking = _resolve_thinking_flag(request)

    try:
        prompt, prompt_opened_think, image_tensors = target_engine.format_chat_prompt(
            raw_messages,
            tools=request.tools,
            enable_thinking=enable_thinking,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to format chat prompt: {e}")

    stop_strings = [request.stop] if isinstance(request.stop, str) else request.stop
    temp = request.temperature if request.temperature is not None else 0.7
    do_sample = temp > 0.0

    # 1. 串流回應模式 (Streaming via SSE)
    if request.stream:
        async def event_generator() -> AsyncGenerator[str, None]:
            first_chunk = ChatCompletionChunk(
                id=req_id,
                model=target_engine.model_name,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=DeltaMessage(role="assistant", content=""),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {first_chunk.model_dump_json(exclude_none=True)}\n\n"

            stream_parser = IncrementalStreamParser(
                prompt_opened_think=prompt_opened_think,
                extract_thinking=True,
                extract_tools=True,
            )
            final_metrics: Optional[EngineMetrics] = None
            reasoning_acc: List[str] = []
            emitted_tool_calls = False

            async for token_text, metrics in target_engine.generate_stream(
                prompt=prompt,
                max_new_tokens=request.max_tokens or 1024,
                temperature=temp,
                top_p=request.top_p or 0.9,
                top_k=request.top_k or 50,
                repetition_penalty=request.repetition_penalty or 1.0,
                stop_strings=stop_strings,
                do_sample=do_sample,
                images=image_tensors if image_tensors else None,
            ):
                if token_text:
                    deltas = stream_parser.feed(token_text)
                    for d in deltas:
                        if d.reasoning_content:
                            reasoning_acc.append(d.reasoning_content)
                            chunk = ChatCompletionChunk(
                                id=req_id,
                                model=target_engine.model_name,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0,
                                        delta=DeltaMessage(reasoning_content=d.reasoning_content),
                                        finish_reason=None,
                                    )
                                ],
                            )
                            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                        if d.content:
                            chunk = ChatCompletionChunk(
                                id=req_id,
                                model=target_engine.model_name,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0,
                                        delta=DeltaMessage(content=d.content),
                                        finish_reason=None,
                                    )
                                ],
                            )
                            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                        if d.tool_calls:
                            emitted_tool_calls = True
                            tc_dicts = [tc.to_openai_dict(i) for i, tc in enumerate(d.tool_calls)]
                            chunk = ChatCompletionChunk(
                                id=req_id,
                                model=target_engine.model_name,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0,
                                        delta=DeltaMessage(tool_calls=tc_dicts),
                                        finish_reason=None,
                                    )
                                ],
                            )
                            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                if metrics:
                    final_metrics = metrics

            # 沖刷串流解析器尾端殘留
            for d in stream_parser.finalize():
                if d.reasoning_content:
                    reasoning_acc.append(d.reasoning_content)
                    chunk = ChatCompletionChunk(
                        id=req_id,
                        model=target_engine.model_name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=DeltaMessage(reasoning_content=d.reasoning_content),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                if d.content:
                    chunk = ChatCompletionChunk(
                        id=req_id,
                        model=target_engine.model_name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=DeltaMessage(content=d.content),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                if d.tool_calls:
                    emitted_tool_calls = True
                    tc_dicts = [tc.to_openai_dict(i) for i, tc in enumerate(d.tool_calls)]
                    chunk = ChatCompletionChunk(
                        id=req_id,
                        model=target_engine.model_name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=DeltaMessage(tool_calls=tc_dicts),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

            usage_info = None
            if final_metrics:
                r_toks = target_engine.count_tokens("".join(reasoning_acc)) if reasoning_acc else 0
                final_metrics.reasoning_tokens = r_toks
                usage_info = _build_usage_info(final_metrics, reasoning_tokens=r_toks)

            finish_reason = "tool_calls" if emitted_tool_calls else "stop"
            stop_chunk = ChatCompletionChunk(
                id=req_id,
                model=target_engine.model_name,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=DeltaMessage(),
                        finish_reason=finish_reason,
                    )
                ],
                usage=usage_info,
            )
            yield f"data: {stop_chunk.model_dump_json(exclude_none=True)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # 2. 同步完整生成模式 (Non-streaming)
    try:
        output_text, metrics = await target_engine.generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens or 1024,
            temperature=temp,
            top_p=request.top_p or 0.9,
            top_k=request.top_k or 50,
            repetition_penalty=request.repetition_penalty or 1.0,
            stop_strings=stop_strings,
            do_sample=do_sample,
            images=image_tensors if image_tensors else None,
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference engine error: {e}")

    parsed = parse_full_output(output_text, prompt_opened_think=prompt_opened_think)
    r_tokens = target_engine.count_tokens(parsed.reasoning_content) if parsed.reasoning_content else 0
    metrics.reasoning_tokens = r_tokens

    tc_list = [tc.to_openai_dict(i) for i, tc in enumerate(parsed.tool_calls)] if parsed.tool_calls else None
    msg = ChatMessage(
        role="assistant",
        content=parsed.content,
        reasoning_content=parsed.reasoning_content,
        tool_calls=tc_list,
    )

    return ChatCompletionResponse(
        id=req_id,
        model=target_engine.model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=msg,
                finish_reason=parsed.finish_reason,
            )
        ],
        usage=_build_usage_info(metrics, reasoning_tokens=r_tokens),
    )


@router.post("/v1/completions")
async def create_completion(
    request: CompletionRequest,
    engine: OVStrataEngine = Depends(get_engine),
):
    target_engine, is_sub = resolve_engine_for_model(request.model)
    req_id = f"cmpl-{uuid.uuid4().hex}"
    prompt_str = request.prompt if isinstance(request.prompt, str) else "\n".join(request.prompt)
    stop_strings = [request.stop] if isinstance(request.stop, str) else request.stop
    temp = request.temperature if request.temperature is not None else 0.7

    output_text, metrics = await target_engine.generate(
        prompt=prompt_str,
        max_new_tokens=request.max_tokens or 512,
        temperature=temp,
        top_p=request.top_p or 0.9,
        repetition_penalty=request.repetition_penalty or 1.0,
        stop_strings=stop_strings,
        do_sample=temp > 0.0,
    )

    return {
        "id": req_id,
        "object": "text_completion",
        "model": target_engine.model_name,
        "choices": [
            {
                "index": 0,
                "text": output_text,
                "finish_reason": "stop",
            }
        ],
        "usage": _build_usage_info(metrics).model_dump(exclude_none=True),
    }


# =====================================================================
# Anthropic Messages API 相容端點 (/v1/messages)
# =====================================================================

@router.post("/v1/messages")
async def create_anthropic_message(
    request: AnthropicMessageRequest,
    engine: OVStrataEngine = Depends(get_engine),
):
    target_engine, is_sub = resolve_engine_for_model(request.model)
    if is_sub:
        logger.info(
            f"Routing Anthropic message [model='{request.model}'] -> SUB-ENGINE "
            f"[{target_engine.current_device}: {target_engine.model_name}]"
        )
    else:
        logger.info(
            f"Routing Anthropic message [model='{request.model}'] -> PRIMARY "
            f"[{target_engine.current_device}: {target_engine.model_name}]"
        )

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    converted_messages: List[Dict[str, Any]] = []

    if request.system:
        if isinstance(request.system, str):
            converted_messages.append({"role": "system", "content": request.system})
        elif isinstance(request.system, list):
            sys_text = "\n".join(
                b.get("text", "") for b in request.system if isinstance(b, dict) and b.get("type") == "text"
            )
            if sys_text:
                converted_messages.append({"role": "system", "content": sys_text})

    for m in request.messages:
        converted_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

    enable_thinking: Optional[bool] = None
    if request.thinking and isinstance(request.thinking, dict):
        t_type = request.thinking.get("type", "").lower()
        if t_type == "enabled":
            enable_thinking = True
        elif t_type == "disabled":
            enable_thinking = False

    prompt, prompt_opened_think, image_tensors = target_engine.format_chat_prompt(
        converted_messages,
        tools=request.tools,
        enable_thinking=enable_thinking,
    )

    temp = request.temperature if request.temperature is not None else 0.7
    do_sample = temp > 0.0

    if request.stream:
        async def anthropic_event_generator() -> AsyncGenerator[str, None]:
            start_evt = {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": target_engine.model_name,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": target_engine.count_tokens(prompt), "output_tokens": 0},
                },
            }
            yield f"event: message_start\ndata: {json.dumps(start_evt, ensure_ascii=False)}\n\n"

            block_start = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
            yield f"event: content_block_start\ndata: {json.dumps(block_start, ensure_ascii=False)}\n\n"

            stream_parser = IncrementalStreamParser(
                prompt_opened_think=prompt_opened_think,
                extract_thinking=True,
                extract_tools=True,
            )
            final_metrics: Optional[EngineMetrics] = None
            emitted_tools: List[Any] = []

            async for token_text, metrics in target_engine.generate_stream(
                prompt=prompt,
                max_new_tokens=request.max_tokens,
                temperature=temp,
                top_p=request.top_p or 0.9,
                top_k=request.top_k or 50,
                stop_strings=request.stop_sequences,
                do_sample=do_sample,
                images=image_tensors if image_tensors else None,
            ):
                if token_text:
                    for d in stream_parser.feed(token_text):
                        if d.content:
                            delta_evt = {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": d.content},
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(delta_evt, ensure_ascii=False)}\n\n"
                        if d.tool_calls:
                            emitted_tools.extend(d.tool_calls)
                if metrics:
                    final_metrics = metrics

            for d in stream_parser.finalize():
                if d.content:
                    delta_evt = {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": d.content},
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(delta_evt, ensure_ascii=False)}\n\n"
                if d.tool_calls:
                    emitted_tools.extend(d.tool_calls)

            block_stop = {"type": "content_block_stop", "index": 0}
            yield f"event: content_block_stop\ndata: {json.dumps(block_stop, ensure_ascii=False)}\n\n"

            for idx, tc in enumerate(emitted_tools, start=1):
                tc_start = {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {"type": "tool_use", "id": tc.id, "name": tc.name, "input": {}},
                }
                yield f"event: content_block_start\ndata: {json.dumps(tc_start, ensure_ascii=False)}\n\n"
                tc_delta = {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "input_json_delta", "partial_json": tc.arguments_json},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(tc_delta, ensure_ascii=False)}\n\n"
                tc_stop = {"type": "content_block_stop", "index": idx}
                yield f"event: content_block_stop\ndata: {json.dumps(tc_stop, ensure_ascii=False)}\n\n"

            stop_reason = "tool_use" if emitted_tools else "end_turn"
            out_toks = final_metrics.completion_tokens if final_metrics else 0
            msg_delta = {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": out_toks},
            }
            yield f"event: message_delta\ndata: {json.dumps(msg_delta, ensure_ascii=False)}\n\n"
            yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

        return StreamingResponse(
            anthropic_event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    output_text, metrics = await target_engine.generate(
        prompt=prompt,
        max_new_tokens=request.max_tokens,
        temperature=temp,
        top_p=request.top_p or 0.9,
        top_k=request.top_k or 50,
        stop_strings=request.stop_sequences,
        do_sample=do_sample,
        images=image_tensors if image_tensors else None,
    )

    parsed = parse_full_output(output_text, prompt_opened_think=prompt_opened_think)
    content_blocks: List[Dict[str, Any]] = []
    if parsed.reasoning_content:
        content_blocks.append({"type": "thinking", "thinking": parsed.reasoning_content})
    if parsed.content or not parsed.tool_calls:
        content_blocks.append({"type": "text", "text": parsed.content})
    for tc in parsed.tool_calls:
        content_blocks.append(tc.to_anthropic_block())

    stop_reason = "tool_use" if parsed.tool_calls else "end_turn"
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": target_engine.model_name,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": metrics.prompt_tokens,
            "output_tokens": metrics.completion_tokens,
            "ttft_seconds": round(metrics.ttft, 4),
            "tokens_per_second": round(metrics.tps, 2),
            "offload_ratio": metrics.offload_ratio,
        },
    }
