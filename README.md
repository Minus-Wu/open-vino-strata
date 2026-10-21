# OpenVINO-Strata — Low-VRAM 35B-A3B MoE OpenAI & Anthropic LLM Provider

**OpenVINO-Strata** 結合 **Intel OpenVINO 2026 (`openvino-genai`)** 硬體加速與 **PolyStrata 分層式記憶與運算架構 (Stratified Memory & Compute Architecture)**，專為在有限 GPU 顯存（如 **6 GB ~ 16 GB VRAM** 的 Intel Arc iGPU / dGPU 與消費級筆電）上流暢運行 **`Qwen3.8-35B-A3B` / `Qwen3.6-35B-A3B` (`qwen35moe`)** 等大型稀疏混合專家模型（35B 總參數、每 Token 僅激活 ~3B 參數）而設計。

---

## ✨ 核心架構亮點 (PolyStrata × OpenVINO 2026)

1. **Strata 專家權重分層卸載 (`OFFLOAD_RATIO`)**
   - 將 `qwen35moe`（40 層：30 層 Gated DeltaNet 線性注意力 + 10 層標準 Full Attention，每層 256 個路由專家、每 Token 激活 8 個專家 + 1 個共享專家）拆分為 **Dense 主幹**（約 3.6 GB）與 **稀疏專家池**（約 14.4 GB）。
   - 依據使用者設定的 `vram_budget_gb` 自動計算最佳 `OFFLOAD_RATIO`（0%~100%），將 Dense 主幹與高頻熱點專家駐留於高速 GPU VRAM，並將冷門專家權重卸載至系統 LPDDR5X/DDR5 RAM。
2. **混合注意力 + KV Cache INT8 量化 (`KV_CACHE_PRECISION="u8"`)**
   - 75% 層為線性注意力（固定維度循環狀態，不隨上下文膨脹），僅 10 層需要標準 KV Cache；搭配 OpenVINO `u8` KV Cache 壓縮，32K 長上下文僅需約 **0.65 GB** 顯存。
3. **Multi-Token Prediction (`MTP`) 內建草稿推測解碼**
   - 自動偵測模型目錄中的 `openvino_mtp_model.xml`，將輕量 MTP 草稿頭配置於 CPU/NPU 協同主模型進行 Speculative Decoding。
4. **完整 `<think>` 推理鏈與 `<tool_call>` 工具呼叫串流狀態機**
   - 支援 `enable_thinking`（當設為 `False` 時自動預填 `<think>\n\n</think>\n\n` 跳過冗長思考以秒回），並在 SSE 串流與非串流模式即時分離 `reasoning_content` 與雙格式 `tool_calls`（JSON 與 Qwen3.6/3.8 `<function=...><parameter=...>` XML 格式）。
5. **雙重 API 相容 + PolyStrata 動態顯存生命週期**
   - 完整相容 **OpenAI API** (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) 與 **Anthropic Messages API** (`/v1/messages`，可直接對接 **Claude Code** / **Cursor** / **Cline**)。
   - 提供 `/api/models/load`、`/api/models/unload` 與 `/api/strata/plan`，支援無縫熱切換模型與一鍵釋放顯存。

---

## 📊 Qwen3.8-35B-A3B / Qwen3.6-35B-A3B 顯存分層對照表

由 `python scripts/check_devices.py` 針對 `qwen35moe` (INT4 + INT8 backup) 實測規劃之顯存配置矩陣：

| VRAM 預算上限 (`vram_budget_gb`) | 專家卸載比例 (`OFFLOAD_RATIO`) | 預估 GPU VRAM 佔用 | 預估系統 RAM 卸載量 | KV Cache 精度 | 適用硬體情境 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **6.0 GB** | `100%` | **5.25 GB** | 15.48 GB | `u8` | 6 GB 入門獨顯 / 嚴格顯存限制 |
| **8.0 GB** | `85%` | **7.29 GB** | 13.44 GB | `u8` | 8 GB 消費級顯卡 / 16 GB 整合顯存筆電 |
| **10.0 GB** | `70%` | **9.32 GB** | 11.41 GB | `u8` | 10~12 GB 中階顯卡 |
| **12.0 GB** | `55%` | **11.36 GB** | 9.37 GB | `u8` | 12 GB 顯卡 / 32 GB 筆電共享顯存 |
| **14.5 GB** | `35%` | **14.07 GB** | 6.66 GB | `u8` | Intel Arc 140V (16GB iGPU, 預設最佳檔位) |
| **16.0 GB** | `25%` | **15.43 GB** | 5.30 GB | `u8` | 16 GB+ 顯卡高吞吐模式 |

---

## 🚀 快速開始 (Quick Start)

### 1. 檢查硬體與 Strata 分層就緒度

```powershell
.\.venv\Scripts\python.exe scripts/check_devices.py
```

### 2. 下載或轉換 `Qwen3.8-35B-A3B` / `Qwen3.6-35B-A3B` 模型

#### 方式 A：直接下載預量化 OpenVINO INT4 IR 模型（推薦，支援 `curl.exe` 斷點續傳）
```powershell
# 查看所有預設支援的 35B-A3B 與輕量測試模型
.\.venv\Scripts\python.exe scripts/download_model.py --list

# 下載官方 OpenVINO INT4 35B-A3B 模型 (~18 GB，含 MTP 與 Vision 子模型)
.\.venv\Scripts\python.exe scripts/download_model.py --preset qwen3.6-35b-a3b-int4-ov

# 若僅需純文字推論，可略過 Vision 子模型節省約 415 MB
.\.venv\Scripts\python.exe scripts/download_model.py --preset qwen3.6-35b-a3b-int4-ov --skip-vision

# 下載社群 Qwen3.8-35B-A3B-Distill GGUF (Q3_K_M / IQ2_M / Q4_K_M)
.\.venv\Scripts\python.exe scripts/download_model.py --preset qwen3.8-35b-a3b-distill-gguf-q3
```

#### 方式 B：將 `empero-ai/Qwen3.8-35B-A3B-Distill` (Safetensors) 轉換為 OpenVINO INT4 混合精度 IR
```powershell
# 預覽轉換指令與混合精度配方 (INT4 group_size=64 + INT8_sym 保護 Router 與 Embeddings)
.\.venv\Scripts\python.exe scripts/convert_model.py --model-id empero-ai/Qwen3.8-35B-A3B-Distill --dry-run

# 執行轉換 (可加上 --install-deps 自動安裝 optimum-intel 與 nncf)
.\.venv\Scripts\python.exe scripts/convert_model.py --model-id empero-ai/Qwen3.8-35B-A3B-Distill --install-deps
```

### 3. 一鍵啟動伺服器與原生 PyQt6 桌面 AI Studio

直接執行專案根目錄的一鍵啟動腳本：
```powershell
.\start_all.bat
```
或分別啟動後端 API 伺服器與桌面介面：
```powershell
# 啟動 API Server (指定顯存預算為 8.0 GB，Strata 將自動配置 OFFLOAD_RATIO=85%)
.\.venv\Scripts\python.exe run_server.py --device GPU --vram-budget 8.0 --port 12223

# 啟動 PyQt6 桌面聊天室
.\.venv\Scripts\python.exe run_gui.py --url http://127.0.0.1:12223
```

---

## 🔌 API 端點一覽

| 方法與路徑 | 相容標準 | 說明 |
| :--- | :---: | :--- |
| `POST /v1/chat/completions` | OpenAI | 支援 SSE 串流、`enable_thinking`、`reasoning_content`、`tools`/`tool_calls` 與 Base64/URL 圖片輸入 |
| `POST /v1/completions` | OpenAI | 傳統文字補全端點 |
| `GET  /v1/models` | OpenAI | 列出所有本地可用模型及其 Strata 顯存預估資訊 |
| `POST /v1/messages` | Anthropic | 完整支援 Anthropic Messages API（可直接供 `Claude Code` 連接使用） |
| `GET  /api/models` | PolyStrata | 查詢所有已發現模型、目前載入狀態與建議卸載比例 |
| `POST /api/models/load` | PolyStrata | 動態熱切換模型或即時變更 `vram_budget_gb` / `offload_ratio` |
| `POST /api/models/unload` | PolyStrata | 立即釋放目前模型的 GPU VRAM 與系統 RAM |
| `POST /api/strata/plan` | PolyStrata | 預先試算任意模型在指定 VRAM 預算下的 `OFFLOAD_RATIO` 與記憶體佔用 |
| `GET  /health` | 系統監控 | 回傳即時 GPU VRAM 分配量、系統可用 RAM 與模型載入狀態 |

---

## 🧪 測試與效能基準量測

```powershell
# 1. 執行單元測試 (Strata 顯存規劃器 + 思考/工具串流解析器 + GPU 引擎生命週期)
.\.venv\Scripts\python.exe scripts/test_engine.py

# 2. 執行 FastAPI 伺服器整合測試 (OpenAI + Anthropic + PolyStrata 動態卸載/重載)
.\.venv\Scripts\python.exe scripts/test_client.py

# 3. 執行吞吐量與不同 OFFLOAD_RATIO 對比基準測試
.\.venv\Scripts\python.exe scripts/benchmark.py --sweep-offload 0,35,55,85 --max-tokens 64
```
