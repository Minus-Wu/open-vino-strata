"""
非同步背景工作執行緒 (QThread Workers for OpenVINO-Strata GUI)
包含：
1. ChatStreamWorker   : 處理 OpenAI 相容 SSE 串流 (`reasoning_content` + `content` + `tool_calls` + `usage`)
2. HealthCheckWorker  : 定期輪詢 `/health` 與 `/api/models` 取得模型載入狀態、Offload Ratio 與 GPU VRAM 即時資訊
3. ModelControlWorker : 背景執行 `/api/models/load` (熱切換模型 / 調整顯存預算) 與 `/api/models/unload` (一鍵釋放顯存)
"""

import json
import time
from typing import List, Dict, Any, Optional

import httpx
from PyQt6.QtCore import QThread, pyqtSignal


class ChatStreamWorker(QThread):
    reasoning_received = pyqtSignal(str)
    token_received = pyqtSignal(str)
    tool_calls_received = pyqtSignal(list)
    metrics_received = pyqtSignal(dict)
    finished_signal = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        base_url: str,
        messages: List[Dict[str, Any]],
        model: str = "default",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_thinking: Optional[bool] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.messages = messages
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking
        self.tools = tools
        self._is_cancelled = False

    def stop(self):
        self._is_cancelled = True

    def run(self):
        url = f"{self.base_url}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": True,
        }
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking
        if self.tools:
            payload["tools"] = self.tools

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        token_count = 0
        server_usage: Optional[Dict[str, Any]] = None

        try:
            with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        err_body = response.read().decode("utf-8", errors="replace")
                        self.error_occurred.emit(f"HTTP {response.status_code}: {err_body}")
                        return

                    for line in response.iter_lines():
                        if self._is_cancelled:
                            break
                        if not line or not line.startswith("data: "):
                            continue

                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            if chunk.get("usage"):
                                server_usage = chunk["usage"]

                            choices = chunk.get("choices", [])
                            if not choices:
                                continue

                            delta = choices[0].get("delta", {})
                            r_text = delta.get("reasoning_content")
                            c_text = delta.get("content")
                            t_calls = delta.get("tool_calls")

                            if r_text:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                token_count += 1
                                self.reasoning_received.emit(r_text)

                            if c_text:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                token_count += 1
                                self.token_received.emit(c_text)

                            if t_calls:
                                self.tool_calls_received.emit(t_calls)
                        except json.JSONDecodeError:
                            continue

            total_time = time.perf_counter() - start_time
            ttft = (first_token_time - start_time) if first_token_time else total_time
            tps = token_count / total_time if total_time > 0 else 0.0

            if server_usage:
                metrics = {
                    "tokens": server_usage.get("completion_tokens", token_count),
                    "reasoning_tokens": server_usage.get("reasoning_tokens", 0),
                    "ttft": round(server_usage.get("ttft_seconds", ttft), 3),
                    "tps": round(
                        server_usage.get("decode_tokens_per_second")
                        or server_usage.get("tokens_per_second")
                        or tps,
                        2,
                    ),
                    "total_time": round(total_time, 2),
                    "offload_ratio": server_usage.get("offload_ratio", 0),
                    "vram_mb": server_usage.get("vram_allocated_mb", 0.0),
                }
            else:
                metrics = {
                    "tokens": token_count,
                    "reasoning_tokens": 0,
                    "ttft": round(ttft, 3),
                    "tps": round(tps, 2),
                    "total_time": round(total_time, 2),
                    "offload_ratio": 0,
                    "vram_mb": 0.0,
                }

            self.metrics_received.emit(metrics)
            self.finished_signal.emit()

        except httpx.ConnectError:
            self.error_occurred.emit(
                f"無法連線至後端伺服器 ({self.base_url})。\n請確認已啟動 run_server.py 或 start_all.bat。"
            )
        except Exception as e:
            self.error_occurred.emit(f"串流推論發生錯誤: {str(e)}")


class HealthCheckWorker(QThread):
    status_updated = pyqtSignal(bool, dict)

    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url.rstrip("/")

    def run(self):
        try:
            with httpx.Client(timeout=3.5) as client:
                resp = client.get(f"{self.base_url}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    models_resp = client.get(f"{self.base_url}/api/models")
                    if models_resp.status_code == 200:
                        data["available_models"] = models_resp.json().get("models", [])
                    self.status_updated.emit(True, data)
                else:
                    self.status_updated.emit(False, {})
        except Exception:
            self.status_updated.emit(False, {})


class ModelControlWorker(QThread):
    completed = pyqtSignal(bool, str, dict)

    def __init__(
        self,
        base_url: str,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.action = action  # "load" or "unload"
        self.payload = payload or {}

    def run(self):
        try:
            with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                if self.action == "unload":
                    resp = client.post(f"{self.base_url}/api/models/unload")
                    if resp.status_code == 200:
                        data = resp.json()
                        self.completed.emit(
                            True,
                            f"已釋放模型 [{data.get('unloaded_model')}] 顯存 (VRAM: {data.get('vram_after_mb', 0):.1f} MB)",
                            data,
                        )
                    else:
                        self.completed.emit(False, f"釋放顯存失敗: {resp.text}", {})
                elif self.action == "load":
                    resp = client.post(f"{self.base_url}/api/models/load", json=self.payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        plan = data.get("strata_plan", {})
                        self.completed.emit(
                            True,
                            f"已載入 [{data.get('model')}] 於 {data.get('device')} "
                            f"(Offload={plan.get('recommended_offload_ratio', 0)}%, 耗時 {data.get('load_time_seconds', 0)}s)",
                            data,
                        )
                    else:
                        self.completed.emit(False, f"載入模型失敗: {resp.text}", {})
        except Exception as e:
            self.completed.emit(False, f"模型控制請求失敗: {e}", {})
