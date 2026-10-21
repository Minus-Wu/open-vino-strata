"""
PolyStrata 風格之思維鏈 (<think>) 與工具呼叫 (<tool_call>) 解析器 (Reasoning & Tool Parser)
支援 Qwen3.8 / Qwen3.6 / Qwen3.5 / DeepSeek-R1 等模型之：
1. <think>...</think> 推論過程分離 (自動映射為 OpenAI `reasoning_content` 與 Anthropic `thinking` 區塊)
2. 雙格式 <tool_call> 解析：
   - 標準 JSON 格式: <tool_call>{"name": "fn", "arguments": {...}}</tool_call>
   - Qwen3.6/3.8 XML 標籤格式: <tool_call><function=fn><parameter=k>v</parameter></function></tool_call>
3. 串流增量狀態機 (IncrementalStreamParser)：防止標籤碎片外洩至一般文字輸出。
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class ParsedToolCall:
    id: str
    name: str
    arguments_json: str
    arguments_dict: Dict[str, Any] = field(default_factory=dict)

    def to_openai_dict(self, index: int = 0) -> Dict[str, Any]:
        return {
            "index": index,
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments_json,
            },
        }

    def to_anthropic_block(self) -> Dict[str, Any]:
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.arguments_dict,
        }


@dataclass
class ParsedOutput:
    content: str = ""
    reasoning_content: Optional[str] = None
    tool_calls: List[ParsedToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


@dataclass
class StreamDelta:
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ParsedToolCall]] = None


# 正則表達式：匹配 XML 風格 <function=NAME><parameter=K>V</parameter>...</function>
_FUNC_XML_RE = re.compile(r"<function=([^>\s]+)>\s*(.*?)\s*(?:</function>|$)", re.DOTALL)
_PARAM_XML_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.DOTALL)


def _parse_single_tool_call_body(body: str) -> Optional[ParsedToolCall]:
    """解析單一 <tool_call> 內部字串，同時支援 JSON 與 Qwen3.6/3.8 XML 格式"""
    raw = body.strip()
    if not raw:
        return None

    call_id = f"call_{uuid.uuid4().hex[:16]}"

    # 1. 嘗試 Qwen3.6 / Qwen3.8 XML 格式: <function=NAME><parameter=P>V</parameter></function>
    m_func = _FUNC_XML_RE.search(raw)
    if m_func:
        fn_name = m_func.group(1).strip()
        params_body = m_func.group(2)
        args_dict: Dict[str, Any] = {}
        for m_param in _PARAM_XML_RE.finditer(params_body):
            p_key = m_param.group(1).strip()
            p_val_str = m_param.group(2).strip()
            # 嘗試解析型別 (int/float/bool/json)，否則保留字串
            try:
                p_val = json.loads(p_val_str)
            except Exception:
                p_val = p_val_str
            args_dict[p_key] = p_val
        return ParsedToolCall(
            id=call_id,
            name=fn_name,
            arguments_json=json.dumps(args_dict, ensure_ascii=False),
            arguments_dict=args_dict,
        )

    # 2. 嘗試標準 JSON 格式: {"name": "...", "arguments": {...}}
    try:
        # 去除可能包裹的 markdown code block
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
        data = json.loads(cleaned)
        if isinstance(data, dict) and "name" in data:
            fn_name = str(data["name"])
            args = data.get("arguments", data.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args_dict = json.loads(args)
                except Exception:
                    args_dict = {"raw": args}
                args_json = args
            elif isinstance(args, dict):
                args_dict = args
                args_json = json.dumps(args, ensure_ascii=False)
            else:
                args_dict = {"value": args}
                args_json = json.dumps(args_dict, ensure_ascii=False)

            return ParsedToolCall(
                id=call_id,
                name=fn_name,
                arguments_json=args_json,
                arguments_dict=args_dict,
            )
    except Exception:
        pass

    return None


def parse_full_output(
    raw_text: str,
    prompt_opened_think: bool = False,
    extract_thinking: bool = True,
    extract_tools: bool = True,
) -> ParsedOutput:
    """
    對完整生成文字進行一次性解析，分離 <think> 思維鏈與 <tool_call> 工具呼叫。
    """
    text = raw_text or ""
    reasoning_parts: List[str] = []
    tool_calls: List[ParsedToolCall] = []

    if extract_thinking:
        # 若 Prompt 結尾已經開啟了 <think>，則第一個 </think> 之前皆為思維鏈
        if prompt_opened_think and "</think>" in text and not text.lstrip().startswith("<think>"):
            think_part, rest = text.split("</think>", 1)
            if think_part.strip():
                reasoning_parts.append(think_part.strip())
            text = rest

        # 擷取所有成對的 <think>...</think>
        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        for m in think_pattern.finditer(text):
            if m.group(1).strip():
                reasoning_parts.append(m.group(1).strip())
        text = think_pattern.sub("", text)

        # 若因達到 max_tokens 而只有開啟 <think> 尚未閉合
        if "<think>" in text:
            before, unclosed_think = text.split("<think>", 1)
            if unclosed_think.strip():
                reasoning_parts.append(unclosed_think.strip())
            text = before

    if extract_tools and "<tool_call>" in text:
        tc_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        for m in tc_pattern.finditer(text):
            parsed_tc = _parse_single_tool_call_body(m.group(1))
            if parsed_tc:
                tool_calls.append(parsed_tc)
        text = tc_pattern.sub("", text)

    content_clean = text.strip() if (reasoning_parts or tool_calls) else text
    reasoning_str = "\n\n".join(reasoning_parts) if reasoning_parts else None
    finish_reason = "tool_calls" if tool_calls else "stop"

    return ParsedOutput(
        content=content_clean,
        reasoning_content=reasoning_str,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
    )


class IncrementalStreamParser:
    """
    串流增量解析狀態機：
    逐塊接收模型輸出的 token 字串，即時區分：
    - 一般內容 (`content`)
    - 思維鏈內容 (`reasoning_content`)
    - 工具呼叫 (`tool_calls`)
    並妥善緩衝可能跨 token 切斷的 `<think>`, `</think>`, `<tool_call>`, `</tool_call>` 標籤。
    """

    TAGS = ("<think>", "</think>", "<tool_call>", "</tool_call>")
    MAX_TAG_LEN = max(len(t) for t in TAGS)

    def __init__(
        self,
        prompt_opened_think: bool = False,
        extract_thinking: bool = True,
        extract_tools: bool = True,
    ):
        self.extract_thinking = extract_thinking
        self.extract_tools = extract_tools
        self.state: str = "THINK" if (prompt_opened_think and extract_thinking) else "CONTENT"
        self.buffer: str = ""
        self.tool_buffer: str = ""
        self.tool_calls: List[ParsedToolCall] = []

    def _might_be_tag_prefix(self, s: str) -> int:
        """
        檢查字串結尾是否可能是某個控制標籤的前綴，回傳需要保留在 buffer 的字元數。
        """
        if not s:
            return 0
        # 只需要檢查最後 MAX_TAG_LEN - 1 個字元內最後出現的 '<'
        tail = s[-(self.MAX_TAG_LEN - 1):]
        idx = tail.rfind("<")
        if idx == -1:
            return 0
        candidate = tail[idx:]
        for tag in self.TAGS:
            if tag.startswith(candidate):
                return len(candidate)
        return 0

    def feed(self, token_text: str) -> List[StreamDelta]:
        """餵入新產出的 token_text，回傳零或多個可立即送出的 StreamDelta"""
        if not token_text:
            return []

        if not self.extract_thinking and not self.extract_tools:
            return [StreamDelta(content=token_text)]

        self.buffer += token_text
        deltas: List[StreamDelta] = []

        while self.buffer:
            if self.state == "CONTENT":
                # 尋找最早出現的 <think> 或 <tool_call>
                candidates: List[Tuple[int, str]] = []
                if self.extract_thinking:
                    i_th = self.buffer.find("<think>")
                    if i_th != -1:
                        candidates.append((i_th, "<think>"))
                if self.extract_tools:
                    i_tc = self.buffer.find("<tool_call>")
                    if i_tc != -1:
                        candidates.append((i_tc, "<tool_call>"))

                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    pos, matched_tag = candidates[0]
                    if pos > 0:
                        deltas.append(StreamDelta(content=self.buffer[:pos]))
                    self.buffer = self.buffer[pos + len(matched_tag):]
                    if matched_tag == "<think>":
                        self.state = "THINK"
                    else:
                        self.state = "TOOL"
                        self.tool_buffer = ""
                    continue

                # 沒有完整標籤，檢查結尾是否為標籤前綴
                keep = self._might_be_tag_prefix(self.buffer)
                emit_len = len(self.buffer) - keep
                if emit_len > 0:
                    deltas.append(StreamDelta(content=self.buffer[:emit_len]))
                    self.buffer = self.buffer[emit_len:]
                break

            elif self.state == "THINK":
                pos = self.buffer.find("</think>")
                if pos != -1:
                    if pos > 0:
                        deltas.append(StreamDelta(reasoning_content=self.buffer[:pos]))
                    self.buffer = self.buffer[pos + len("</think>"):]
                    # 去除 </think> 後緊接著的前導換行
                    self.buffer = self.buffer.lstrip("\r\n")
                    self.state = "CONTENT"
                    continue

                keep = self._might_be_tag_prefix(self.buffer)
                emit_len = len(self.buffer) - keep
                if emit_len > 0:
                    deltas.append(StreamDelta(reasoning_content=self.buffer[:emit_len]))
                    self.buffer = self.buffer[emit_len:]
                break

            elif self.state == "TOOL":
                pos = self.buffer.find("</tool_call>")
                if pos != -1:
                    raw_tool = self.buffer[:pos]
                    self.buffer = self.buffer[pos + len("</tool_call>"):]
                    parsed_tc = _parse_single_tool_call_body(raw_tool)
                    if parsed_tc:
                        self.tool_calls.append(parsed_tc)
                        deltas.append(StreamDelta(tool_calls=[parsed_tc]))
                    self.state = "CONTENT"
                    continue
                else:
                    # 尚未遇到完整的 </tool_call>，保留於 buffer 等待後續 token
                    break

        return deltas

    def finalize(self) -> List[StreamDelta]:
        """串流結束時沖刷殘留 buffer"""
        deltas: List[StreamDelta] = []
        if self.buffer:
            if self.state == "THINK":
                deltas.append(StreamDelta(reasoning_content=self.buffer))
            elif self.state == "CONTENT":
                deltas.append(StreamDelta(content=self.buffer))
            elif self.state == "TOOL":
                raw_tool = self.buffer.replace("</tool_call>", "").strip()
                parsed_tc = _parse_single_tool_call_body(raw_tool)
                if parsed_tc:
                    self.tool_calls.append(parsed_tc)
                    deltas.append(StreamDelta(tool_calls=[parsed_tc]))
            self.buffer = ""

        return deltas


def format_tools_system_instruction(tools: Optional[List[Dict[str, Any]]]) -> str:
    """
    將 OpenAI 或 Anthropic 格式的 tools 清單轉為 Qwen3 / Qwen3.6 / Qwen3.8 標準工具呼叫系統提示詞，
    確保即使透過自訂 Prompt 模板亦能精準觸發 `<tool_call>`。
    """
    if not tools:
        return ""

    normalized_tools = []
    for t in tools:
        if "function" in t:
            normalized_tools.append(t)
        elif "name" in t and "input_schema" in t:
            # Anthropic tool format -> OpenAI tool format
            normalized_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["input_schema"],
                },
            })
        else:
            normalized_tools.append(t)

    tools_json = "\n".join(json.dumps(t, ensure_ascii=False) for t in normalized_tools)
    return (
        "\n\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{tools_json}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    )
