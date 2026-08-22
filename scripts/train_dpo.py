"""
DPO 训练脚本（LoRA，在 SFT 模型基础上做偏好对齐）

用法：
    冒烟：python scripts/train_dpo.py --smoke
    全量：python scripts/train_dpo.py --base output/sft_model_merged --output output/dpo_model

数据：data/dpo_pairs.jsonl（{"prompt","chosen","rejected"}，trl DPOTrainer 原生格式）
说明：
- 从 SFT 合并模型继续训练（保持 SFT 学到的能力，只做偏好对齐）
- LoRA 配置与 SFT 一致（r16/α32/all-linear）
- DPO 学习率明显低于 SFT（1e-5 量级）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent


def load_pairs(path: Path, limit: int | None = None) -> list[dict]:
    items = [
        json.loads(l)
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    if limit:
        items = items[:limit]
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="output/sft_model_merged", help="SFT 合并模型")
    ap.add_argument("--data", default="data/dpo_pairs.jsonl")
    ap.add_argument("--output", default="output/dpo_model")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 对（冒烟）")
    ap.add_argument("--max-steps", type=int, default=None, help="限制总步数（冒烟）")
    ap.add_argument("--qlora", action="store_true",
                    help="4bit 量化加载（QLoRA），8B 在 24GB 显存下的稳妥选项")
    ap.add_argument("--smoke", action="store_true", help="冒烟模式")
    args = ap.parse_args()

    if args.smoke:
        args.limit = args.limit or 40
        args.max_steps = args.max_steps or 10
        args.batch = 2
        args.accum = 2
        args.epochs = 1
        args.output = "output/dpo_smoke"

    # 1. 数据
    print("[1/5] 加载 DPO 数据 ...", flush=True)
    pairs = load_pairs(ROOT / args.data, args.limit)
    ds = Dataset.from_list(pairs)
    print(f"[1/5] {len(ds)} 对", flush=True)

    # 2. 模型（从 SFT 模型继续）
    print("[2/5] 加载 SFT 模型 ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / args.base))
    if args.qlora:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(ROOT / args.base),
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="sdpa",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(ROOT / args.base),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
    # workaround: trl 0.24 DPOTrainer 在 __init__ 访问 model.warnings_issued（transformers Trainer 属性），
    # peft 包装后 __getattr__ 代理到 base model 找不到该属性而报错；预置实例属性规避
    model.warnings_issued = {}
    print("[2/5] 模型加载完成", flush=True)

    # 3. LoRA 配置（通过 peft_config 传给 DPOTrainer，由 trl 内部包装，避免兼容问题）
    print("[3/5] 配置 LoRA ...", flush=True)
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    # 4. DPO 配置
    print("[4/5] 配置 DPOTrainer ...", flush=True)
    from trl import DPOConfig, DPOTrainer

    output_dir = str(ROOT / args.output)
    dpo_args = dict(
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        bf16=True,
        gradient_checkpointing=False,
        max_length=args.max_len,
        max_prompt_length=512,
        logging_strategy="steps",
        logging_steps=5,
        report_to="none",
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        save_only_model=True,
    )
    if args.max_steps:
        dpo_args["max_steps"] = args.max_steps
    config = DPOConfig(**dpo_args)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # 未指定时 trl 使用当前模型作为参考模型
        args=config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # 5. 训练
    print("[5/5] 开始 DPO 训练 ...", flush=True)
    trainer.train()
    trainer.save_model(output_dir)
    print(f"✅ DPO 完成，adapter 已保存到: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
