"""
SFT 训练脚本（LoRA）

用法：
    冒烟测试（验证流程）：python scripts/train_sft.py --smoke
    完整训练（本地 0.6B）：python scripts/train_sft.py --model model/Qwen3-0.6B --output output/sft_model
    完整训练（云上 8B）  ：python scripts/train_sft.py --model /path/Qwen3-8B --output output/sft_model_8b

说明：
- 数据格式：trl messages 格式（data/sft_train.jsonl）
- Qwen3 的 assistant_only_loss 需要自定义聊天模板（new_chat_template.jinja）
- 输出：LoRA adapter（peft 格式），后续用 merge_model.py 合并回基座
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

# 强制离线，避免任何 HuggingFace hub 网络访问
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CHAT_TEMPLATE = ROOT / "new_chat_template.jinja"


def load_messages_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    """纯 Python 读取 messages 格式 JSONL（绕开 datasets.load_dataset 的 hub/多进程行为）。"""
    items = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit:
        items = items[:limit]
    return items


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/Qwen3-0.6B")
    ap.add_argument("--data", default="data/sft_train.jsonl")
    ap.add_argument("--eval-data", default="data/sft_test.jsonl")
    ap.add_argument("--output", default="output/sft_model")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 条训练数据")
    ap.add_argument("--limit-eval", type=int, default=None, help="只用前 N 条评估数据")
    ap.add_argument("--max-steps", type=int, default=None, help="限制总步数（冒烟用）")
    ap.add_argument("--no-eval", action="store_true", help="跳过评估（冒烟快速验证）")
    ap.add_argument("--smoke", action="store_true", help="冒烟模式：200 条 + 20 步 + 小 batch")
    ap.add_argument("--report-to", choices=["none", "tensorboard"], default="none",
                    help="日志上报方式；tensorboard 时日志写入 logs/sft")
    ap.add_argument("--qlora", action="store_true",
                    help="4bit 量化加载（QLoRA），8B 在 24GB 显存下的稳妥选项")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke:
        args.limit = args.limit or 200
        args.limit_eval = args.limit_eval or 50
        args.max_steps = args.max_steps or 20
        args.batch = 2
        args.accum = 2
        args.lr = 2e-4
        args.epochs = 1
        args.report_to = "none"
        args.output = args.output if args.output != "output/sft_model" else "output/sft_smoke"

    # tensorboard 日志目录（trl/transformers 通过环境变量指定）
    if args.report_to == "tensorboard":
        tb_dir = ROOT / "logs" / "sft"
        tb_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(tb_dir)
        print(f"[0/6] tensorboard 日志目录: {tb_dir}", flush=True)

    # 1. 加载数据（已是 messages 格式，纯 Python 读取避免 load_dataset 卡死）
    print("[1/6] 加载数据 ...", flush=True)
    train_items = load_messages_jsonl(ROOT / args.data, args.limit)
    train_ds = Dataset.from_list(train_items)
    eval_ds = None
    if not args.no_eval:
        eval_items = load_messages_jsonl(ROOT / args.eval_data, args.limit_eval)
        eval_ds = Dataset.from_list(eval_items)
    print(f"[1/6] 数据就绪: 训练 {len(train_ds)} 条"
          + (f"，评估 {len(eval_ds)} 条" if not args.no_eval else "，无评估"), flush=True)

    # 2. 加载模型与 tokenizer
    print("[2/6] 加载 tokenizer ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / args.model))
    print("[2/6] 加载模型 ...", flush=True)
    if args.qlora:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(ROOT / args.model),
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="sdpa",
        )
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(ROOT / args.model),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
    print(f"[2/6] 模型已加载: {args.model}，参数 {model.num_parameters()/1e6:.0f}M", flush=True)

    # 3. LoRA 配置
    print("[3/6] 配置 LoRA ...", flush=True)
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. SFT 训练配置
    print("[4/6] 配置 SFTTrainer ...", flush=True)
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer

    output_dir = str(ROOT / args.output)
    train_args = dict(
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        bf16=True,
        gradient_checkpointing=False,
        max_length=args.max_len,
        logging_strategy="steps",
        logging_steps=5,
        report_to=args.report_to,
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        save_only_model=True,
        # Qwen3 原生模板与 assistant_only_loss 不兼容，使用修改后的模板
        assistant_only_loss=True,
        chat_template_path=str(CHAT_TEMPLATE),
    )
    if args.max_steps:
        train_args["max_steps"] = args.max_steps
    if not args.no_eval:
        train_args.update(
            eval_strategy="steps",
            eval_steps=10,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
        )
    config = SFTConfig(**train_args)

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # 5. 训练
    print("[5/6] 开始训练 ...", flush=True)
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[6/6] ✅ SFT 完成，adapter 已保存到: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
