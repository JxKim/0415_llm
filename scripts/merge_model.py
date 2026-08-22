"""
合并 LoRA adapter 到基座模型（避免推理时的额外开销）

用法：
    python scripts/merge_model.py --base model/Qwen3-0.6B --adapter output/sft_smoke --output output/sft_smoke_merged

说明：
- 合并后输出标准模型目录（权重 + tokenizer），可直接被 evalscope / vLLM / transformers 加载
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="model/Qwen3-0.6B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    base_path = str(ROOT / args.base)
    adapter_path = str(ROOT / args.adapter)
    output_path = str(ROOT / args.output)

    print(f"[1/3] 加载基座模型: {base_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    print(f"[2/3] 加载 adapter: {adapter_path}", flush=True)
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    merged = peft_model.merge_and_unload()

    print(f"[3/3] 保存合并模型: {output_path}", flush=True)
    merged.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"✅ 合并完成: {output_path}")


if __name__ == "__main__":
    main()
