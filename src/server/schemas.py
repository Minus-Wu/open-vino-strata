"""
OpenAI / Anthropic / PolyStrata 相容 API 規範資料結構定義 (Pydantic Models)
支援：
- /v1/chat/completions (含 reasoning_content、tool_calls、多模態 image_url 與 SSE 串流)
- /v1/completions
- /v1/models
- /v1/messages (Anthropic Messages API 相容端點，支援 Claude Code / Cline 等工具)
- /api/models, /api/models/load, /api/models/unload, /api/strata/plan (PolyStrata 風格動態模型與顯存分層管理)
"""

import time
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = ""
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    model: Optional[str] = "default"
    messages: List[ChatMessage]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    max_tokens: Optional[int] = Field(default=1024, alias="max_completion_tokens")
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = 1.0
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    reasoning_effort: Optional[str] = None
    enable_thinking: Optional[bool] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = "default"
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = 1.0


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    # Strata 擴展效能與顯存指標
    ttft_seconds: Optional[float] = None
    prefill_tokens_per_second: Optional[float] = None
    tokens_per_second: Optional[float] = None
    decode_tokens_per_second: Optional[float] = None
    offload_ratio: Optional[int] = None
    vram_allocated_mb: Optional[float] = None
    ram_used_gb: Optional[float] = None
    mtp_enabled: Optional[bool] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]
    usage: Optional[UsageInfo] = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "openvino-strata"
    device: Optional[str] = None
    loaded: Optional[bool] = None
    offload_ratio: Optional[int] = None
    estimated_vram_gb: Optional[float] = None
    device_info: Optional[Dict[str, Any]] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]


class AnthropicMessageRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "default"
    messages: List[Dict[str, Any]]
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    max_tokens: int = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    thinking: Optional[Dict[str, Any]] = None


class ModelLoadRequest(BaseModel):
    model: str
    device: Optional[str] = None
    offload_ratio: Optional[int] = None
    vram_budget_gb: Optional[float] = None
    kv_cache_precision: Optional[str] = None
    enable_mtp: Optional[bool] = None
    mtp_device: Optional[str] = None
    enable_vision: Optional[bool] = None


class StrataPlanRequest(BaseModel):
    model: str = "models/Qwen3.8-35B-A3B-int4-ov"
    device: str = "GPU"
    offload_ratio: Optional[int] = None
    vram_budget_gb: Optional[float] = None
    vram_reserve_mib: int = 2048
    kv_cache_precision: str = "u8"
    enable_mtp: bool = False
    mtp_device: str = "CPU"
