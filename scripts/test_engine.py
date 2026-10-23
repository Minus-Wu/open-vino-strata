"""
OpenVINO-Strata 核心引擎與分層顯存規劃單元整合測試 (Engine & Strata Planner Test)
驗證：
1. StrataMemoryPlanner 針對 Qwen3.8-35B-A3B 於 8GB / 12GB / 14.5GB 顯存預算下的 OFFLOAD_RATIO 計算
2. ReasoningAndToolParser 針對 <think> 思維鏈與雙格式 <tool_call> (JSON & Qwen3.8 XML) 的同步與串流解析
3. OVStrataEngine 的模型載入、串流推論、動態顯存釋放 (unload_model) 與自動喚醒 (ensure_loaded)
"""

import asyncio
import sys
import io
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.engine.device_manager import device_manager, StrataConfig
from src.engine.parser import parse_full_output, IncrementalStreamParser
from src.engine.ov_strata_engine import OVStrataEngine


def test_strata_memory_planner():
    print("=" * 65)
    print("[Test 1] 測試 StrataMemoryPlanner (Qwen3.8-35B-A3B 顯存分層規劃)")
    print("=" * 65)

    for budget in [8.0, 12.0, 14.5]:
        cfg = StrataConfig(device="GPU", vram_budget_gb=budget, kv_cache_precision="u8")
        plan = device_manager.plan_strata_execution("models/Qwen3.8-35B-A3B-int4-ov", cfg)
        print(
            f"  - 目標顯存預算 {budget:>4.1f} GB => "
            f"建議 OFFLOAD_RATIO={plan.recommended_offload_ratio:>2d}% | "
            f"駐留熱點專家={plan.hot_experts_in_vram_pct:>2d}% | "
            f"預計 VRAM={plan.estimated_vram_usage_gb:>4.2f} GB | "
            f"預計 RAM 卸載={plan.estimated_ram_offload_gb:>5.2f} GB"
        )
        assert 0 <= plan.recommended_offload_ratio <= 100
        assert plan.estimated_vram_usage_gb <= budget + 0.5

    # 驗證 8GB 預算之卸載比例應高於 14.5GB 預算
    plan_8g = device_manager.plan_strata_execution(
        "models/Qwen3.8-35B-A3B-int4-ov", StrataConfig(vram_budget_gb=8.0)
    )
    plan_14g = device_manager.plan_strata_execution(
        "models/Qwen3.8-35B-A3B-int4-ov", StrataConfig(vram_budget_gb=14.5)
    )
    assert plan_8g.recommended_offload_ratio > plan_14g.recommended_offload_ratio

    # 驗證副模型 RAM 預留功能
    plan_with_sub = device_manager.plan_strata_execution(
        "models/Qwen3.8-35B-A3B-int4-ov",
        StrataConfig(vram_budget_gb=8.0, sub_model_ram_budget_gb=2.0),
    )
    assert plan_with_sub.sub_model_ram_budget_gb == 2.0
    assert plan_with_sub.estimated_ram_offload_gb >= plan_8g.estimated_ram_offload_gb + 1.9
    print(f"  - 副模型預留 2.0 GB RAM 後預計系統 RAM 總用量: {plan_with_sub.estimated_ram_offload_gb} GB")
    print("[V] StrataMemoryPlanner 測試通過！\n")


def test_reasoning_and_tool_parser():
    print("=" * 65)
    print("[Test 2] 測試 <think> 思維鏈與 <tool_call> 解析器 (同步與串流)")
    print("=" * 65)

    sample_xml_output = (
        "<think>\n使用者需要查詢台北天氣，我應該呼叫 get_weather 工具。\n</think>\n"
        "好的，我來為您查詢：\n"
        "<tool_call><function=get_weather><parameter=city>Taipei</parameter><parameter=unit>celsius</parameter></function></tool_call>"
    )
    parsed = parse_full_output(sample_xml_output)
    assert parsed.reasoning_content == "使用者需要查詢台北天氣，我應該呼叫 get_weather 工具。"
    assert "好的，我來為您查詢：" in parsed.content
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "get_weather"
    assert parsed.tool_calls[0].arguments_dict == {"city": "Taipei", "unit": "celsius"}
    assert parsed.finish_reason == "tool_calls"

    # 測試串流跨碎片切分
    chunks = [
        "<thi", "nk>正在分析", "問題...</th", "ink>\n這是回", "覆內容<tool_",
        'call>{"name": "calc", "arguments": {"x": 42}}</tool', "_call>"
    ]
    sp = IncrementalStreamParser()
    stream_reasoning = ""
    stream_content = ""
    stream_tools = []
    for c in chunks:
        for d in sp.feed(c):
            if d.reasoning_content:
                stream_reasoning += d.reasoning_content
            if d.content:
                stream_content += d.content
            if d.tool_calls:
                stream_tools.extend(d.tool_calls)
    for d in sp.finalize():
        if d.reasoning_content:
            stream_reasoning += d.reasoning_content
        if d.content:
            stream_content += d.content
        if d.tool_calls:
            stream_tools.extend(d.tool_calls)

    assert stream_reasoning == "正在分析問題..."
    assert stream_content == "這是回覆內容"
    assert len(stream_tools) == 1 and stream_tools[0].name == "calc"
    print("[V] 同步與串流思維鏈/工具解析器測試通過！\n")


async def test_ov_strata_engine_runtime():
    print("=" * 65)
    print("[Test 3] 測試 OVStrataEngine 實機 GPU 分層推論與動態顯存釋放")
    print("=" * 65)

    engine = OVStrataEngine(auto_load=False)
    models = engine.discover_available_models()
    print(f"  - 探索到可用模型數量: {len(models)}")
    for m in models:
        print(f"    * {m['id']} ({m['total_weight_gb']} GB, 預估 VRAM: {m['estimated_vram_gb']} GB)")

    if not models:
        print("[!] 尚未下載任何實體模型，略過實機推論測試。")
        return

    # 選取最小的模型進行快速驗證
    models_sorted = sorted(models, key=lambda x: x["total_weight_gb"])
    test_model = models_sorted[0]["id"]
    target_dev = "GPU" if device_manager.is_device_available("GPU") else "CPU"

    cfg = StrataConfig(device=target_dev, offload_ratio=20, kv_cache_precision="u8")
    switch_res = await engine.switch_model(test_model, device=target_dev, strata_config=cfg)
    print(f"  - 成功載入模型: {switch_res['model']} 於 [{switch_res['device']}] (耗時 {switch_res['load_time_seconds']}s)")

    prompt, opened_think, _ = engine.format_chat_prompt(
        [{"role": "user", "content": "請用一句話回答：1+1等於多少？"}],
        enable_thinking=False,
    )

    pieces = []
    final_metrics = None
    async for tok, m in engine.generate_stream(prompt, max_new_tokens=32, do_sample=False):
        if tok:
            pieces.append(tok)
        if m:
            final_metrics = m

    reply = "".join(pieces).strip()
    print(f"  - 串流推論結果: {reply}")
    assert len(reply) > 0
    if final_metrics:
        print(
            f"  - 效能指標: TTFT={final_metrics.ttft:.3f}s | "
            f"TPS={final_metrics.tps:.2f} tok/s | "
            f"Offload={final_metrics.offload_ratio}%"
        )

    # 測試動態卸載與自動喚醒
    unload_res = await engine.unload_model()
    print(f"  - 動態釋放顯存狀態: {unload_res['status']} (釋放模型: {unload_res['unloaded_model']})")
    assert not engine.is_loaded

    text2, m2 = await engine.generate(prompt, max_new_tokens=16, do_sample=False)
    print(f"  - 惰性自動喚醒推論成功 (Loaded={engine.is_loaded}): {text2.strip()[:40]}...")
    assert engine.is_loaded
    print("[V] OVStrataEngine 實機測試全部通過！\n")


if __name__ == "__main__":
    test_strata_memory_planner()
    test_reasoning_and_tool_parser()
    asyncio.run(test_ov_strata_engine_runtime())
