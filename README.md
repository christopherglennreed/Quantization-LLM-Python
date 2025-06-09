#!/usr/bin/env python3
"""
Quantise an HF checkpoint with either GPTQ (W4A16 / W8A16) or FP8-dynamic.

Examples
--------
# 4-bit Mixtral on two GPUs
python compressor.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
                     --bits 4 --gpus 2 --out mixtral-w4a16

# FP8 Llama-3-70B on four GPUs
python compressor.py --model meta-llama/Llama-3.3-70B-Instruct \
                     --bits fp8 --gpus 4 --out llama3-fp8
"""
from pathlib import Path
import argparse, re, torch

# ── imports that work across llmcompressor versions ─────────────────────────
try:
    from llmcompressor.transformers import AutoModelForCausalLM
except ImportError:  # ≤0.5.1 wheels
    from llmcompressor.transformers import SparseAutoModelForCausalLM as AutoModelForCausalLM

from llmcompressor.transformers import oneshot
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier

# save_quant_config exists only ≥0.16; fall back to a tiny writer if absent
try:
    from llmcompressor.transformers import save_quant_config
except ImportError:
    def save_quant_config(model, output_dir):
        import json, safetensors.torch as st, glob
        scheme = {}
        for f in glob.glob(str(Path(model) / "*.safetensors")):
            with st.safe_open(f, framework="pt") as sf:
                for k in sf.keys():
                    if "W4A16" in k:
                        scheme[k] = "w4"
                    elif "W8A16" in k:
                        scheme[k] = "w8"
                    elif "FP8" in k.upper():
                        scheme[k] = "fp8"
        qcfg = {
            "target_scheme_map": scheme,
            "ignore": [],
            "quant_format": "compressed_tensors",
            "sparsity_scheme_map": {},
            "sparsity_ignore_list": [],
        }
        Path(output_dir, "quantization_config.json").write_text(json.dumps(qcfg, indent=2))
# ────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF repo-id, local folder, or *.safetensors file")
    p.add_argument("--out", default="compressed_model")
    p.add_argument("--bits",
                   help="4, 8, or 'fp8' for FP8-dynamic",
                   choices=("4", "8", "fp8"), default="4")
    p.add_argument("--gpus", type=int, default=2,
                   help="Number of GPUs used for sharding the FP16 model")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.model).expanduser()
    out = Path(args.out).expanduser().resolve()


    
    device_map = calculate_offload_device_map(
        src, num_gpus=args.gpus, reserve_for_hessians=True, torch_dtype="auto"
    )

    model = AutoModelForCausalLM.from_pretrained(
        src, device_map=device_map, torch_dtype="auto"
    )

    # choose recipe
    if args.bits == "fp8":
        recipe = [QuantizationModifier(
            targets="Linear",
            scheme="FP8_dynamic",
            ignore=["lm_head", r".*experts\.\d+\.(w1|w2)\.weight", r".*router.*weight"],
        )]
    else:
        recipe = [GPTQModifier(
            scheme=f"W{args.bits}A16",
            targets="Linear",
            ignore=[r".*experts\.\d+\.(w1|w2)\.weight",
                    r".*router.*weight",
                    "lm_head"],
            group_size=128,
            damp_percent=0.02,
            sym=True,
        )]

    oneshot(
        model=model,
        dataset="open_platypus",
        recipe=recipe,
        output_dir=out,
        max_seq_length=1024,
        num_calibration_samples=128,
    )

    # make sure vLLM has the manifest
    save_quant_config(model=out, output_dir=out)
    print(f"✅  Quantised model saved to {out}")


if __name__ == "__main__":
    main()
