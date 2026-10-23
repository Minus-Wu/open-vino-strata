"""
OpenVINO-Strata 模型轉換與混合精度量化工具 (scripts/convert_model.py)
支援雙模式：
1. 【低記憶體串流直轉模式 (Direct Streaming Quantizer, < 2 GB RAM)】：
   針對 `Qwen3.8-35B-A3B-Distill` / `Qwen3.6-35B-A3B` (`qwen3_5_moe` 架構)，當本機已有同架構之 OpenVINO IR 模板
   (`models/Qwen3.6-35B-A3B-int4-ov`) 時，直接以逐張量串流方式從 Safetensors 讀取權重，即時執行 NNCF 等價之
   INT4_ASYM (group_size=128) + INT8_SYM/ASYM 混合精度量化並寫入 `.bin`，完全免除 PyTorch 全模型載入所需的 80GB+ 記憶體！
2. 【標準 Optimum-CLI 導出模式】：
   針對其他一般模型，呼叫 `optimum-cli export openvino` 進行完整 PyTorch 追蹤與 NNCF 壓縮。
"""

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.download_model import download_repo


def parse_ov_xml_constants(xml_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """解析 OpenVINO .xml 中的所有 Const 節點及其第一個下游消費節點 (Consumer)"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    layers_elem = root.find("layers")
    edges_elem = root.find("edges")

    id_to_layer = {}
    for node in layers_elem:
        id_to_layer[node.attrib["id"]] = node

    from_to_consumers: Dict[str, List[str]] = {}
    for edge in edges_elem:
        src_id = edge.attrib["from-layer"]
        dst_id = edge.attrib["to-layer"]
        dst_name = id_to_layer[dst_id].attrib.get("name", "")
        from_to_consumers.setdefault(src_id, []).append(dst_name)

    by_name: Dict[str, Dict[str, Any]] = {}
    by_consumer: Dict[str, Dict[str, Any]] = {}

    for lid, node in id_to_layer.items():
        if node.attrib.get("type") != "Const":
            continue
        data_elem = node.find("data")
        if data_elem is None:
            continue
        attr = data_elem.attrib
        shape_str = attr.get("shape", "").strip()
        shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip()) if shape_str else ()
        info = {
            "id": lid,
            "name": node.attrib.get("name", ""),
            "element_type": attr.get("element_type", ""),
            "offset": int(attr.get("offset", 0)),
            "size": int(attr.get("size", 0)),
            "shape": shape,
        }
        by_name[info["name"]] = info
        for cons in from_to_consumers.get(lid, []):
            by_consumer[cons] = info

    return by_name, by_consumer


def _pack_u4(arr_u8: np.ndarray) -> bytes:
    """將 [0..15] 的 uint8 陣列依 OpenVINO u4 規範打包 (偶數索引在低 4 bits，奇數索引在高 4 bits)"""
    flat = arr_u8.reshape(-1, 2)
    packed = (flat[:, 0] & 0x0F) | ((flat[:, 1] & 0x0F) << 4)
    return packed.astype(np.uint8).tobytes()


def write_quantized_weight(
    f_bin,
    by_name: Dict[str, Dict[str, Any]],
    prefix: str,
    weight_f32: np.ndarray,
):
    """將單一權重張量量化為 OpenVINO IR 指定的 u4 / u8 / i8 格式並寫入 .bin 對應 offset"""
    w_info = by_name[prefix]
    sc_info = by_name[prefix + "/scale"]
    zp_info = by_name.get(prefix + "/zero_point")

    w = weight_f32.reshape(w_info["shape"])
    el_type = w_info["element_type"]

    if el_type == "u4":
        w_min = np.minimum(w.min(axis=-1, keepdims=True), 0.0)
        w_max = np.maximum(w.max(axis=-1, keepdims=True), 0.0)
        scale_f16 = ((w_max - w_min) / 15.0).astype(np.float16)
        scale_f32 = np.where(scale_f16 == 0, 1.0, scale_f16.astype(np.float32))
        zp_u8 = np.clip(np.round(-w_min / scale_f32), 0, 15).astype(np.uint8)
        w_u8 = np.clip(np.round(w / scale_f32 + zp_u8), 0, 15).astype(np.uint8)

        w_bytes = _pack_u4(w_u8)
        zp_bytes = _pack_u4(zp_u8)
        sc_bytes = scale_f16.tobytes()

        assert len(w_bytes) == w_info["size"], f"{prefix} size mismatch: {len(w_bytes)} != {w_info['size']}"
        assert zp_info and len(zp_bytes) == zp_info["size"], f"{prefix}/zp size mismatch"
        assert len(sc_bytes) == sc_info["size"], f"{prefix}/scale size mismatch"

        f_bin.seek(w_info["offset"])
        f_bin.write(w_bytes)
        f_bin.seek(zp_info["offset"])
        f_bin.write(zp_bytes)
        f_bin.seek(sc_info["offset"])
        f_bin.write(sc_bytes)

    elif el_type == "u8":
        w_min = np.minimum(w.min(axis=-1, keepdims=True), 0.0)
        w_max = np.maximum(w.max(axis=-1, keepdims=True), 0.0)
        scale_f16 = ((w_max - w_min) / 255.0).astype(np.float16)
        scale_f32 = np.where(scale_f16 == 0, 1.0, scale_f16.astype(np.float32))
        zp_u8 = np.clip(np.round(-w_min / scale_f32), 0, 255).astype(np.uint8)
        w_u8 = np.clip(np.round(w / scale_f32 + zp_u8), 0, 255).astype(np.uint8)

        w_bytes = w_u8.tobytes()
        zp_bytes = zp_u8.tobytes()
        sc_bytes = scale_f16.tobytes()

        assert len(w_bytes) == w_info["size"]
        assert zp_info and len(zp_bytes) == zp_info["size"]
        assert len(sc_bytes) == sc_info["size"]

        f_bin.seek(w_info["offset"])
        f_bin.write(w_bytes)
        f_bin.seek(zp_info["offset"])
        f_bin.write(zp_bytes)
        f_bin.seek(sc_info["offset"])
        f_bin.write(sc_bytes)

    elif el_type == "i8":
        abs_max = np.maximum(np.max(np.abs(w), axis=-1, keepdims=True), 1e-8)
        scale_f16 = (abs_max / 127.0).astype(np.float16)
        scale_f32 = np.where(scale_f16 == 0, 1.0, scale_f16.astype(np.float32))
        w_i8 = np.clip(np.round(w / scale_f32), -127, 127).astype(np.int8)

        w_bytes = w_i8.tobytes()
        sc_bytes = scale_f16.tobytes()

        assert len(w_bytes) == w_info["size"]
        assert len(sc_bytes) == sc_info["size"]

        f_bin.seek(w_info["offset"])
        f_bin.write(w_bytes)
        f_bin.seek(sc_info["offset"])
        f_bin.write(sc_bytes)
    else:
        raise ValueError(f"Unsupported quantized element_type: {el_type} for {prefix}")


def write_f32_const(f_bin, const_info: Dict[str, Any], arr_f32: np.ndarray):
    raw = arr_f32.astype(np.float32).tobytes()
    assert len(raw) == const_info["size"], (
        f"f32 const {const_info['name']} size mismatch: {len(raw)} != {const_info['size']}"
    )
    f_bin.seek(const_info["offset"])
    f_bin.write(raw)


def direct_stream_quantize_qwen35moe(
    source_dir: Path,
    template_dir: Path,
    output_dir: Path,
) -> int:
    """
    使用極低記憶體 (< 2GB RAM) 直接將 Qwen3.8-35B-A3B-Distill Safetensors 串流轉換為 OpenVINO INT4 IR
    """
    from safetensors import safe_open

    t0 = time.perf_counter()
    print("=" * 84)
    print(" [Low-RAM Streaming Mode] Converting Qwen3.8-35B-A3B Safetensors -> OpenVINO INT4 IR")
    print(f"   Source Safetensors : {source_dir}")
    print(f"   IR Graph Template  : {template_dir}")
    print(f"   Target Output Dir  : {output_dir}")
    print("=" * 84)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 複製非巨型 .bin 的所有靜態結構與設定檔
    print("\n[1/4] Preparing OpenVINO IR graph files, tokenizer, and vision submodels...")
    for item in template_dir.iterdir():
        if item.is_file() and item.name not in (
            "openvino_language_model.bin",
            "openvino_text_embeddings_model.bin",
            "openvino_mtp_model.bin",
        ):
            shutil.copy2(item, output_dir / item.name)

    for cfg_file in ("chat_template.jinja", "generation_config.json", "README.md"):
        src_cfg = source_dir / cfg_file
        if src_cfg.exists():
            shutil.copy2(src_cfg, output_dir / cfg_file)

    # 2. 轉換 Text Embeddings (openvino_text_embeddings_model.bin)
    print("\n[2/4] Quantizing Text Embeddings (openvino_text_embeddings_model.bin)...")
    emb_xml = output_dir / "openvino_text_embeddings_model.xml"
    emb_bin = output_dir / "openvino_text_embeddings_model.bin"
    shutil.copy2(template_dir / "openvino_text_embeddings_model.bin", emb_bin)
    emb_by_name, _ = parse_ov_xml_constants(emb_xml)

    index_data = json.loads((source_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: Dict[str, str] = index_data["weight_map"]

    emb_shard = source_dir / weight_map["model.language_model.embed_tokens.weight"]
    with safe_open(str(emb_shard), framework="pt") as st, open(emb_bin, "r+b") as f_emb:
        w_emb = st.get_tensor("model.language_model.embed_tokens.weight").float().numpy()
        write_quantized_weight(f_emb, emb_by_name, "self._openvino_orig_weight", w_emb)
        del w_emb
    gc.collect()
    print("  -> Text Embeddings INT8_SYM quantization complete.")

    # 3. 轉換 Multi-Token Prediction 草稿模型 (openvino_mtp_model.bin)
    mtp_src = source_dir / "model-mtp.safetensors"
    mtp_xml = output_dir / "openvino_mtp_model.xml"
    mtp_bin = output_dir / "openvino_mtp_model.bin"
    if mtp_src.exists() and mtp_xml.exists():
        print("\n[3/4] Quantizing Multi-Token Prediction Submodel (openvino_mtp_model.bin)...")
        shutil.copy2(template_dir / "openvino_mtp_model.bin", mtp_bin)
        mtp_by_name, mtp_by_cons = parse_ov_xml_constants(mtp_xml)

        mtp_quant_map = {
            "mtp.fc.weight": "self.fc._openvino_orig_weight",
            "mtp.layers.0.self_attn.q_proj.weight": "self.layers.0.self_attn.q_proj._openvino_orig_weight",
            "mtp.layers.0.self_attn.k_proj.weight": "self.layers.0.self_attn.k_proj._openvino_orig_weight",
            "mtp.layers.0.self_attn.v_proj.weight": "self.layers.0.self_attn.v_proj._openvino_orig_weight",
            "mtp.layers.0.self_attn.o_proj.weight": "self.layers.0.self_attn.o_proj._openvino_orig_weight",
            "mtp.layers.0.mlp.shared_expert_gate.weight": "self.layers.0.mlp.shared_expert_gate._openvino_orig_weight",
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight": "self.layers.0.mlp.shared_expert.gate_proj._openvino_orig_weight",
            "mtp.layers.0.mlp.shared_expert.up_proj.weight": "self.layers.0.mlp.shared_expert.up_proj._openvino_orig_weight",
            "mtp.layers.0.mlp.shared_expert.down_proj.weight": "self.layers.0.mlp.shared_expert.down_proj._openvino_orig_weight",
            "mtp.layers.0.mlp.experts.down_proj": "self.layers.0.mlp.experts.down_proj",
            "mtp.layers.0.mlp.gate.weight": "self.layers.0.mlp.gate.weight",
        }
        mtp_norm_map = {
            "mtp.pre_fc_norm_embedding.weight": "__module.pre_fc_norm_embedding/aten::mul/Multiply_1",
            "mtp.pre_fc_norm_hidden.weight": "__module.pre_fc_norm_hidden/aten::mul/Multiply_1",
            "mtp.layers.0.input_layernorm.weight": "__module.layers.0.input_layernorm/aten::mul/Multiply_1",
            "mtp.layers.0.self_attn.k_norm.weight": "__module.layers.0.self_attn.k_norm/aten::mul/Multiply_1",
            "mtp.layers.0.self_attn.q_norm.weight": "__module.layers.0.self_attn.q_norm/aten::mul/Multiply_1",
            "mtp.layers.0.post_attention_layernorm.weight": "__module.layers.0.post_attention_layernorm/aten::mul/Multiply_1",
            "mtp.norm.weight": "__module.norm/aten::type_as/ConvertLike",
        }

        with safe_open(str(mtp_src), framework="pt") as st, open(mtp_bin, "r+b") as f_mtp:
            for k in st.keys():
                w = st.get_tensor(k).float().numpy()
                if k in mtp_quant_map:
                    write_quantized_weight(f_mtp, mtp_by_name, mtp_quant_map[k], w)
                elif k == "mtp.layers.0.mlp.experts.gate_up_proj":
                    write_quantized_weight(
                        f_mtp, mtp_by_name, "__module.layers.0.mlp/prim::ListUnpack/VariadicSplit.0", w[:, :512, :]
                    )
                    write_quantized_weight(
                        f_mtp, mtp_by_name, "__module.layers.0.mlp/prim::ListUnpack/VariadicSplit.1", w[:, 512:, :]
                    )
                elif k in mtp_norm_map:
                    write_f32_const(f_mtp, mtp_by_cons[mtp_norm_map[k]], 1.0 + w)
                del w
        gc.collect()
        print("  -> MTP submodel INT4_ASYM quantization complete.")

    # 4. 串流轉換主語言模型 (openvino_language_model.bin, 40 層 256 專家)
    print("\n[4/4] Streaming & Quantizing 40-Layer MoE Language Model (openvino_language_model.bin)...")
    lang_xml = output_dir / "openvino_language_model.xml"
    lang_bin = output_dir / "openvino_language_model.bin"
    print("  -> Allocating 16.75 GB binary container from template...")
    shutil.copy2(template_dir / "openvino_language_model.bin", lang_bin)

    print("  -> Parsing XML constant offset table...")
    lang_by_name, lang_by_cons = parse_ov_xml_constants(lang_xml)

    # 依分片分組，確保每個 .safetensors 分片只開啟並讀取一次
    shard_to_keys: Dict[str, List[str]] = {}
    for k, shard_name in weight_map.items():
        if k.startswith("model.visual.") or k.startswith("mtp.") or k == "model.language_model.embed_tokens.weight":
            continue
        shard_to_keys.setdefault(shard_name, []).append(k)

    total_shards = len(shard_to_keys)
    updated_tensors = 0

    with open(lang_bin, "r+b") as f_lang:
        for idx, (shard_name, keys) in enumerate(sorted(shard_to_keys.items()), start=1):
            shard_path = source_dir / shard_name
            t_s = time.perf_counter()
            with safe_open(str(shard_path), framework="pt") as st:
                for k in keys:
                    w = st.get_tensor(k).float().numpy()

                    if k == "lm_head.weight":
                        write_quantized_weight(f_lang, lang_by_name, "self.model.lm_head._openvino_orig_weight", w)
                        updated_tensors += 1
                    elif k == "model.language_model.norm.weight":
                        cons = "__module.model.model.language_model.norm/aten::mul/Multiply_1"
                        write_f32_const(f_lang, lang_by_cons[cons], 1.0 + w)
                        updated_tensors += 1
                    else:
                        m = re.match(r"^model\.language_model\.layers\.(\d+)\.(.+)$", k)
                        if not m:
                            del w
                            continue
                        lid, suffix = m.group(1), m.group(2)
                        base_mod = f"model.model.language_model.layers.{lid}"

                        if suffix == "input_layernorm.weight":
                            write_f32_const(
                                f_lang, lang_by_cons[f"__module.{base_mod}.input_layernorm/aten::mul/Multiply_1"], 1.0 + w
                            )
                        elif suffix == "post_attention_layernorm.weight":
                            write_f32_const(
                                f_lang,
                                lang_by_cons[f"__module.{base_mod}.post_attention_layernorm/aten::mul/Multiply_1"],
                                1.0 + w,
                            )
                        elif suffix == "linear_attn.norm.weight":
                            write_f32_const(
                                f_lang, lang_by_cons[f"__module.{base_mod}.linear_attn.norm/aten::mul/Multiply_1"], w
                            )
                        elif suffix == "linear_attn.A_log":
                            write_f32_const(
                                f_lang, lang_by_name[f"__module.{base_mod}.linear_attn/aten::exp/Exp"], np.exp(w)
                            )
                        elif suffix == "linear_attn.dt_bias":
                            write_f32_const(
                                f_lang, lang_by_cons[f"__module.{base_mod}.linear_attn/aten::add/Add_2"], w
                            )
                        elif suffix == "linear_attn.conv1d.weight":
                            write_quantized_weight(
                                f_lang, lang_by_name, f"__module.{base_mod}.linear_attn/aten::_convolution/Reshape", w
                            )
                        elif suffix in (
                            "linear_attn.in_proj_qkv.weight",
                            "linear_attn.in_proj_a.weight",
                            "linear_attn.in_proj_b.weight",
                            "linear_attn.in_proj_z.weight",
                            "linear_attn.out_proj.weight",
                            "self_attn.q_proj.weight",
                            "self_attn.k_proj.weight",
                            "self_attn.v_proj.weight",
                            "self_attn.o_proj.weight",
                            "mlp.shared_expert_gate.weight",
                            "mlp.shared_expert.gate_proj.weight",
                            "mlp.shared_expert.up_proj.weight",
                            "mlp.shared_expert.down_proj.weight",
                        ):
                            proj_path = suffix[:-7]  # strip .weight
                            write_quantized_weight(
                                f_lang, lang_by_name, f"self.{base_mod}.{proj_path}._openvino_orig_weight", w
                            )
                        elif suffix == "self_attn.q_norm.weight":
                            write_f32_const(
                                f_lang, lang_by_cons[f"__module.{base_mod}.self_attn.q_norm/aten::mul/Multiply_1"], 1.0 + w
                            )
                        elif suffix == "self_attn.k_norm.weight":
                            write_f32_const(
                                f_lang, lang_by_cons[f"__module.{base_mod}.self_attn.k_norm/aten::mul/Multiply_1"], 1.0 + w
                            )
                        elif suffix == "mlp.experts.gate_up_proj":
                            write_quantized_weight(
                                f_lang,
                                lang_by_name,
                                f"__module.{base_mod}.mlp/prim::ListUnpack/VariadicSplit.0",
                                w[:, :512, :],
                            )
                            write_quantized_weight(
                                f_lang,
                                lang_by_name,
                                f"__module.{base_mod}.mlp/prim::ListUnpack/VariadicSplit.1",
                                w[:, 512:, :],
                            )
                        elif suffix == "mlp.experts.down_proj":
                            write_quantized_weight(f_lang, lang_by_name, f"self.{base_mod}.mlp.experts.down_proj", w)
                        elif suffix == "mlp.gate.weight":
                            write_quantized_weight(f_lang, lang_by_name, f"self.{base_mod}.mlp.gate.weight", w)
                        else:
                            raise KeyError(f"Unmapped language model tensor key: {k}")

                        updated_tensors += 1
                    del w
            gc.collect()
            print(
                f"  [{idx:02d}/{total_shards:02d}] Quantized {shard_name} ({len(keys)} tensors) in {time.perf_counter()-t_s:.1f}s"
            )

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 84)
    print(f" Direct Streaming Quantization Complete in {elapsed:.1f}s! ({updated_tensors} language tensors updated)")
    print(f" Output OpenVINO INT4 Model : {output_dir}")
    print("=" * 84)
    return 0


def build_optimum_command(
    model_source: str,
    output_dir: Path,
    weight_format: str = "int4",
    group_size: int = 64,
    ratio: float = 1.0,
    backup_precision: str = "int8_sym",
    sym: bool = False,
    task: str = "auto",
) -> List[str]:
    python_exe = Path(sys.executable)
    optimum_cli = python_exe.parent / ("optimum-cli.exe" if sys.platform == "win32" else "optimum-cli")

    if optimum_cli.exists():
        cmd = [str(optimum_cli), "export", "openvino"]
    else:
        cmd = [str(python_exe), "-m", "optimum.exporters.openvino"]

    resolved_task = task
    if resolved_task == "auto":
        if "35b-a3b" in model_source.lower() or "qwen3.8" in model_source.lower() or "qwen3.6" in model_source.lower():
            resolved_task = "image-text-to-text"
        else:
            resolved_task = "text-generation-with-past"

    cmd.extend([
        "--model", model_source,
        "--task", resolved_task,
        "--weight-format", weight_format,
        "--group-size", str(group_size),
        "--ratio", str(ratio),
        "--backup-precision", backup_precision,
        "--trust-remote-code",
    ])
    if sym:
        cmd.append("--sym")

    cmd.append(str(output_dir))
    return cmd


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Convert Qwen3.8-35B-A3B-Distill / Qwen3.6-35B-A3B to OpenVINO INT4 mixed-precision IR."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="empero-ai/Qwen3.8-35B-A3B-Distill",
        help="HuggingFace model ID or local path (default: empero-ai/Qwen3.8-35B-A3B-Distill)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for OpenVINO IR (default: models/<model-name>-int4-ov)",
    )
    parser.add_argument(
        "--template-dir",
        type=str,
        default=str(ROOT_DIR / "models" / "Qwen3.6-35B-A3B-int4-ov"),
        help="Reference OpenVINO IR template directory for low-RAM streaming quantization",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(ROOT_DIR / ".hf_cache"),
        help="Local download staging directory on D: drive (default: ./.hf_cache)",
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="Delete temporary .hf_cache (66 GB Safetensors) after successful OpenVINO IR export",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Compatibility flag (no extra dependencies needed in direct streaming mode)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the conversion plan without executing",
    )
    args = parser.parse_args()

    model_short_name = Path(args.model_id).name
    out_dir = Path(args.output_dir) if args.output_dir else (ROOT_DIR / "models" / f"{model_short_name}-int4-ov")
    hf_cache_dir = Path(args.cache_dir).resolve()
    is_local_path = Path(args.model_id).exists()
    local_source_dir = Path(args.model_id).resolve() if is_local_path else (hf_cache_dir / model_short_name)
    template_dir = Path(args.template_dir).resolve()

    if args.dry_run:
        print(f"Source: {args.model_id} -> Staging: {local_source_dir} -> Output: {out_dir}")
        print(f"Low-RAM Streaming Template available: {template_dir.exists()} ({template_dir})")
        return 0

    # Step 1: 若尚未完整下載，先透過 curl.exe 斷點續傳下載至 .hf_cache/<model_name>
    if not is_local_path and not (local_source_dir / "model-00021-of-00021.safetensors").exists():
        print(f"\n[Step 1/2] Downloading {args.model_id} to {local_source_dir} (resumable via curl.exe)...")
        dl_rc = download_repo(repo_id=args.model_id, target_dir=local_source_dir)
        if dl_rc != 0:
            return dl_rc

    # Step 2: 若存在同架構 OpenVINO IR 模板，優先執行低記憶體 (< 2GB RAM) 串流直轉
    if template_dir.exists() and (template_dir / "openvino_language_model.xml").exists():
        rc = direct_stream_quantize_qwen35moe(
            source_dir=local_source_dir,
            template_dir=template_dir,
            output_dir=out_dir,
        )
    else:
        cmd = build_optimum_command(str(local_source_dir), out_dir)
        rc = subprocess.run(cmd).returncode

    if rc == 0 and args.clean_cache and not is_local_path and local_source_dir.exists():
        print(f"\n[Clean-Cache] Removing temporary 66 GB source files at {local_source_dir} ...")
        shutil.rmtree(local_source_dir, ignore_errors=True)

    return rc


if __name__ == "__main__":
    sys.exit(main())
