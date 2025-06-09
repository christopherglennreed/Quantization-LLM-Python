Examples
--------
Main goal is Quantize your LLMs. I have been able to get a 70b Model on 80GB or a 32B on a 18gb video card. Good Luck

# 4-bit Mixtral on two GPUs
python compressor.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
                     --bits 4 --gpus 2 --out mixtral-w4a16

# FP8 Llama-3-70B on four GPUs
python compressor.py --model meta-llama/Llama-3.3-70B-Instruct \
                     --bits fp8 --gpus 4 --out llama3-fp8
"""
from pathlib import Path
import argparse, re, torch
