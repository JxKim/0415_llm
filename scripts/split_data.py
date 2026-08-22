"""
数据切分脚本：按种子分组，把生成的变体切分为训练集/测试集，并生成评估集。

输入：data/generated/category{N}_{name}.jsonl（每条含 messages + seed_idx）
输出：
    data/sft_train.jsonl                  训练集（messages 格式，4 类合并，去掉 seed_idx）
    data/sft_test.jsonl                   测试集（messages + category，4 类合并）
    data/eval/{categoryN}_{name}.jsonl    评估集（evalscope general_qa 格式，按类别分文件）

切分策略：每个种子（seed_idx）的变体组内按 9:1 切分
    - 每组 20 条 → 18 训练 + 2 测试；不足 20 按比例，至少留 1 条测试
    - 固定随机种子，保证可复现

用法：python scripts/split_data.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "data" / "generated"
OUT_DIR = ROOT / "data"
EVAL_DIR = ROOT / "data" / "eval"

CATEGORY_FILES = [
    ("category1_知识问答.jsonl", "知识问答", "category1_knowledge_qa"),
    ("category2_故障诊断.jsonl", "故障诊断", "category2_diagnosis"),
    ("category3_话术生成.jsonl", "话术生成", "category3_scripts"),
    ("category4_投诉处理.jsonl", "投诉处理", "category4_complaints"),
]
TEST_PER_GROUP = 2  # 每组留 2 条进测试集


def load_items(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    rng = random.Random(42)  # 固定种子，可复现
    train_items: list[dict] = []
    test_items: list[dict] = []
    eval_dir = EVAL_DIR
    eval_dir.mkdir(parents=True, exist_ok=True)

    for fname, cat_name, subset_name in CATEGORY_FILES:
        path = GEN_DIR / fname
        if not path.exists():
            print(f"[warn] 缺少文件: {path}")
            continue
        items = load_items(path)

        # 按 seed_idx 分组
        groups: dict[int, list[dict]] = defaultdict(list)
        for it in items:
            groups[it.get("seed_idx", 0)].append(it)

        # 每组内切分
        cat_train, cat_test = [], []
        for idx in sorted(groups):
            group = groups[idx]
            rng.shuffle(group)
            n_test = min(TEST_PER_GROUP, max(1, len(group) // 10))
            n_test = min(n_test, len(group))
            tests = group[:n_test]
            trains = group[n_test:]
            for t in trains:
                t = dict(t)
                t.pop("seed_idx", None)
                cat_train.append(t)
            for t in tests:
                t = dict(t)
                t.pop("seed_idx", None)
                t["category"] = cat_name
                cat_test.append(t)

        # 评估集（general_qa 格式，按类别分文件）
        eval_path = eval_dir / f"{subset_name}.jsonl"
        with eval_path.open("w", encoding="utf-8") as f:
            for t in cat_test:
                f.write(json.dumps(
                    {"query": t["messages"][0]["content"]}, ensure_ascii=False
                ) + "\n")

        train_items.extend(cat_train)
        test_items.extend(cat_test)
        print(f"{cat_name}: 生成 {len(items)} → 训练 {len(cat_train)} / 测试 {len(cat_test)}")

    # 写出合并文件
    with (OUT_DIR / "sft_train.jsonl").open("w", encoding="utf-8") as f:
        for it in train_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with (OUT_DIR / "sft_test.jsonl").open("w", encoding="utf-8") as f:
        for it in test_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"\n总计: 训练 {len(train_items)} 条 + 测试 {len(test_items)} 条")
    print(f"输出: data/sft_train.jsonl, data/sft_test.jsonl, data/eval/*.jsonl")


if __name__ == "__main__":
    main()
