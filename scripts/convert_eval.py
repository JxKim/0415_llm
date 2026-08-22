# -*- coding: utf-8 -*-
"""
生成带标准答案（answer）的评估集（LOG-018 优化）

从 data/sft_test.jsonl（含 messages + category）提取每条 query 的标准回复，
生成 {"query": ..., "answer": ...} 格式，输出到 data/eval/ 和 benchmark/custom_eval/text/qa/
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# subset 文件名 ↔ sft_test 中的 category 中文名（与 split_data.py 一致）
CATEGORY_MAP = [
    ("category1_knowledge_qa", "知识问答"),
    ("category2_diagnosis", "故障诊断"),
    ("category3_scripts", "话术生成"),
    ("category4_complaints", "投诉处理"),
]


def main() -> None:
    tests = [
        json.loads(l)
        for l in (ROOT / "data" / "sft_test.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for t in tests:
        by_cat[t.get("category", "未知")].append(t)

    total = 0
    for subset, cat_name in CATEGORY_MAP:
        items = by_cat.get(cat_name, [])
        records = [
            {
                "query": it["messages"][0]["content"],
                "answer": it["messages"][1]["content"],
            }
            for it in items
        ]
        for d in (ROOT / "data" / "eval", ROOT / "benchmark" / "custom_eval" / "text" / "qa"):
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{subset}.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                encoding="utf-8",
            )
        total += len(records)
        print(f"{subset}: {len(records)} 条（含标准答案）")

    print(f"合计: {total} 条，已写入 data/eval/ 与 benchmark/custom_eval/text/qa/")


if __name__ == "__main__":
    main()
