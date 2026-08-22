"""
DPO 偏好对构造脚本（阶段 A：LLM 改写坏回复）

原理：取 SFT 数据的 (user, assistant)，user 作 prompt、assistant 作 chosen，
按类别的【缺陷维度】让 LLM 把 chosen 刻意改写为 rejected。

用法：
    python scripts/build_dpo.py --mode small   # 每类 5 对，验货
    python scripts/build_dpo.py --mode full    # 1000 对（200/200/300/300）

输出：data/dpo_pairs.jsonl，每行 {"prompt", "chosen", "rejected"}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_data  # noqa: E402  复用 call_llm / get_api_key

ROOT = Path(__file__).resolve().parent.parent

# 各类别的缺陷维度（rejected 制造方向，与评估维度对齐）
DEFECTS = {
    1: ["漏关键条件", "价格乱报", "语气冷淡", "不促行动"],
    2: ["紧急程度误判", "建议不安全", "无到店检测建议", "信息缺失"],
    3: ["不道歉", "只拒绝不给方案", "语气生硬", "无促行动"],
    4: ["辩解推诿", "无共情", "无补救", "无升级路径"],
}

GENERATED_FILES = {
    1: "data/generated/category1_知识问答.jsonl",
    2: "data/generated/category2_故障诊断.jsonl",
    3: "data/generated/category3_话术生成.jsonl",
    4: "data/generated/category4_投诉处理.jsonl",
}

FULL_TARGETS = {1: 200, 2: 200, 3: 300, 4: 300}

REWRITE_PROMPT = """你是汽车售后客服领域的资深数据工程师。请把下面的【标准回复】改写成一条【有缺陷的回复】，用于偏好数据构造（DPO 训练）。

要求：
1. 缺陷类型：{defect}。**只在这一方面破坏标准回复**，其余内容保持合理、符合客服身份。
2. 改写后的回复必须和标准回复有清晰差别，让评测者一眼能看出这个缺陷。
3. 保持口语自然，不要直接照抄标准回复，重新组织语言；不要输出 JSON、解释或多余内容。
4. 输出只包含改写后的回复文本。

【客户问题】
{user}

【标准回复】
{assistant}

【缺陷类型】
{defect}

改写后的回复："""


def rewrite_rejected(user: str, chosen: str, defect: str) -> str:
    prompt = REWRITE_PROMPT.format(user=user, assistant=chosen, defect=defect)
    content = build_data.call_llm(prompt, max_tokens=512, temperature=0.8, json_mode=False)
    rejected = content.strip()
    # 清理可能出现的 think 块
    if "</think>" in rejected:
        rejected = rejected.split("</think>")[-1].strip()
    return rejected


def validate(user: str, chosen: str, rejected: str) -> bool:
    if not rejected or len(rejected) < 20 or len(rejected) > 400:
        return False
    if rejected == chosen:
        return False
    return True


def load_items(cat: int) -> list[dict]:
    path = ROOT / GENERATED_FILES[cat]
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["small", "full"], default="small")
    ap.add_argument("--category", type=str, default=None,
                    help="类别 1|2|3|4，支持逗号分隔；默认全部")
    ap.add_argument("--defect-per-item", type=int, default=1,
                    help="每条样本改写的缺陷数（默认 1）")
    args = ap.parse_args()

    rng = random.Random(42)
    categories = [int(c) for c in args.category.split(",")] if args.category else [1, 2, 3, 4]
    targets = FULL_TARGETS if args.mode == "full" else {1: 5, 2: 5, 3: 5, 4: 5}

    out_path = ROOT / "data" / "dpo_pairs.jsonl"
    if args.mode == "small":
        out_path = ROOT / "data" / "dpo_pairs_small.jsonl"
    if out_path.exists():
        out_path.unlink()

    total_ok = 0
    for cat in categories:
        items = load_items(cat)
        # 需要的样本数 = 目标对数 / 每条缺陷数
        need = max(1, targets.get(cat, 5) // args.defect_per_item)
        sample = items[:need] if args.mode == "small" else rng.sample(items, min(need, len(items)))
        print(f"--- 类别{cat}: 样本 {len(sample)} 条（目标 {targets.get(cat, 5)} 对）---")

        ok = 0
        with out_path.open("a", encoding="utf-8") as f:
            for i, it in enumerate(sample, 1):
                user = it["messages"][0]["content"]
                chosen = it["messages"][1]["content"]
                defects = rng.sample(DEFECTS[cat], args.defect_per_item)
                for defect in defects:
                    try:
                        rejected = rewrite_rejected(user, chosen, defect)
                    except Exception as e:  # noqa: BLE001
                        print(f"  [warn] 改写失败: {e}")
                        continue
                    if validate(user, chosen, rejected):
                        f.write(json.dumps(
                            {"prompt": user, "chosen": chosen, "rejected": rejected},
                            ensure_ascii=False,
                        ) + "\n")
                        ok += 1
                    time.sleep(0.3)
                if i % 10 == 0:
                    print(f"  {i}/{len(sample)} 条处理中，已生成 {ok} 对")
        total_ok += ok
        print(f"  ✅ 类别{cat}: {ok} 对")

    print(f"\n总计: {total_ok} 对 → {out_path}")


if __name__ == "__main__":
    main()
