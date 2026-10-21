"""
Stage 2 Integration Test for OpenVINO-Strata FastAPI Server.
Tests:
  1. GET  /health & GET /v1/models
  2. POST /api/strata/plan (PolyStrata memory planning endpoint for 35B-A3B)
  3. GET  /api/models & GET /api/models/current
  4. POST /v1/chat/completions (Non-streaming & SSE Streaming)
  5. POST /v1/messages (Anthropic Messages API non-streaming & streaming)
  6. POST /api/models/unload & POST /api/models/load (Dynamic VRAM lifecycle)
"""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient

from src.engine.device_manager import StrataConfig
from src.server.app import create_app


def find_test_model() -> str:
    candidates = [
        Path(r"D:\Code\aio-mono\open-vino-llm\models\Qwen2.5-Coder-0.5B-Instruct-int4-ov"),
        ROOT_DIR / "models" / "Qwen2.5-Coder-0.5B-Instruct-int4-ov",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError("Could not find POC test model Qwen2.5-Coder-0.5B-Instruct-int4-ov")


def main() -> int:
    print("=" * 72)
    print(" OpenVINO-Strata Stage 2: FastAPI Server Integration Tests")
    print("=" * 72)

    model_dir = find_test_model()
    strata_cfg = StrataConfig(
        device="GPU",
        vram_budget_gb=12.0,
        kv_cache_precision="u8",
        enable_mtp=False,
    )
    app = create_app(
        model_path=model_dir,
        device="GPU",
        strata_config=strata_cfg,
    )

    with TestClient(app) as client:
        # 1. Health check
        print("\n[1/7] Testing GET /health ...")
        res = client.get("/health")
        assert res.status_code == 200, res.text
        health = res.json()
        print(f"  Status: {health['status']}, Model: {health['model']}, Loaded: {health['loaded']}")
        assert health["status"] == "healthy"
        assert health["loaded"] is True
        model_name = health["model"]

        # 2. OpenAI /v1/models & PolyStrata /api/models
        print("\n[2/7] Testing GET /v1/models & GET /api/models ...")
        res = client.get("/v1/models")
        assert res.status_code == 200, res.text
        models_data = res.json()
        print(f"  Found {len(models_data['data'])} models: {[m['id'] for m in models_data['data']]}")
        assert len(models_data["data"]) >= 1

        res_api = client.get("/api/models")
        assert res_api.status_code == 200, res_api.text
        assert res_api.json()["current"] == model_name

        # 3. PolyStrata POST /api/strata/plan (simulating Qwen3.8-35B-A3B on 8 GB VRAM)
        print("\n[3/7] Testing POST /api/strata/plan (Qwen3.8-35B-A3B @ 8GB VRAM) ...")
        res = client.post(
            "/api/strata/plan",
            json={
                "model": "Qwen3.8-35B-A3B-int4-ov",
                "device": "GPU",
                "vram_budget_gb": 8.0,
                "kv_cache_precision": "u8",
                "enable_mtp": True,
            },
        )
        assert res.status_code == 200, res.text
        plan_data = res.json()
        print(
            f"  8GB Plan -> Offload Ratio: {plan_data['recommended_offload_ratio']}%, "
            f"Est VRAM: {plan_data['estimated_vram_usage_gb']} GB, "
            f"Est RAM Offload: {plan_data['estimated_ram_offload_gb']} GB"
        )
        assert plan_data["recommended_offload_ratio"] >= 70

        # 4. Non-streaming /v1/chat/completions
        print("\n[4/7] Testing POST /v1/chat/completions (non-streaming) ...")
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            "max_tokens": 32,
            "temperature": 0.0,
            "stream": False,
        }
        res = client.post("/v1/chat/completions", json=payload)
        assert res.status_code == 200, res.text
        chat_res = res.json()
        reply = chat_res["choices"][0]["message"]["content"]
        usage = chat_res["usage"]
        print(f"  Reply: {reply!r}")
        print(
            f"  Usage: {usage['completion_tokens']} tokens @ {usage.get('tokens_per_second')} tok/s "
            f"(Decode: {usage.get('decode_tokens_per_second')} tok/s, Offload={usage.get('offload_ratio')}%)"
        )
        assert "4" in reply

        # 5. Streaming /v1/chat/completions
        print("\n[5/7] Testing POST /v1/chat/completions (SSE streaming) ...")
        payload["stream"] = True
        chunks = []
        with client.stream("POST", "/v1/chat/completions", json=payload) as stream_res:
            assert stream_res.status_code == 200
            for line in stream_res.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):].strip()
                if data_str == "[DONE]":
                    break
                chunk_obj = json.loads(data_str)
                choices = chunk_obj.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("content"):
                        chunks.append(delta["content"])
        streamed_text = "".join(chunks)
        print(f"  Streamed Reply ({len(chunks)} chunks): {streamed_text!r}")
        assert "4" in streamed_text

        # 6. Anthropic /v1/messages
        print("\n[6/7] Testing POST /v1/messages (Anthropic Messages API) ...")
        anth_payload = {
            "model": model_name,
            "system": "Be concise.",
            "messages": [
                {"role": "user", "content": "What is 3 + 4? Reply with the number."}
            ],
            "max_tokens": 16,
            "temperature": 0.0,
            "stream": False,
        }
        res = client.post("/v1/messages", json=anth_payload)
        assert res.status_code == 200, res.text
        anth_data = res.json()
        print(f"  Anthropic Reply: {anth_data['content'][0]['text']!r}")
        assert anth_data["type"] == "message"
        assert "7" in anth_data["content"][0]["text"]

        # 7. PolyStrata Dynamic Unload & Reload
        print("\n[7/7] Testing POST /api/models/unload & /api/models/load ...")
        res = client.post("/api/models/unload")
        assert res.status_code == 200, res.text
        unload_data = res.json()
        assert unload_data["status"] == "unloaded"
        print(f"  Unloaded model successfully: {unload_data}")

        res_health = client.get("/health")
        assert res_health.json()["loaded"] is False

        res = client.post(
            "/api/models/load",
            json={
                "model": model_name,
                "device": "GPU",
                "vram_budget_gb": 10.0,
                "kv_cache_precision": "u8",
            },
        )
        assert res.status_code == 200, res.text
        load_data = res.json()
        assert load_data["status"] == "loaded"
        print(f"  Reloaded model: {load_data['model']} on {load_data['device']}")

    print("\n" + "=" * 72)
    print(" ALL STAGE 2 SERVER TESTS PASSED SUCCESSFULLY!")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
