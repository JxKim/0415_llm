# -*- coding: utf-8 -*-
"""
生成基线（Prompt Engineering）评估集：给每个 query 注入系统提示词

输出：benchmark/custom_eval/text/qa_baseline/{subset}.jsonl
格式：{"messages": [{"role":"system",...},{"role":"user","content":query}], "answer": 标准答案}
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASELINE_SYSTEM_PROMPT = """你是某汽车品牌 4S 店的售后客服助手，负责解答车主关于保养、维修、质保、配件价格、道路救援、投诉等问题。

回答要求：
1. 语气礼貌、专业、有同理心；不推诿、不敷衍、不辩解。
2. 涉及政策、价格的信息要准确，说明适用条件（如"6个月或5000公里内，先到为准"），价格给合理区间并注明"以门店实际为准"。
3. 涉及车辆故障或安全隐患时，必须先判断紧急程度：高（立即停车、呼叫拖车、禁止继续行驶）、中（尽快进店检测）、低（可预约常规检查），并给出保守安全的建议，禁止建议带故障继续行驶。
4. 处理投诉时遵循：先共情道歉、不辩解 → 界定责任（若确认是我们的责任）→ 给出具体补救方案 → 提供升级渠道（如400服务监督热线）。
5. 回复结构清晰、简洁，给出可执行的后续动作（到店检测/预约/拖车等）。"""

SUBSETS = ["category1_knowledge_qa", "category2_diagnosis", "category3_scripts", "category4_complaints"]


def main() -> None:
    src_dir = ROOT / "benchmark" / "custom_eval" / "text" / "qa"
    out_dir = ROOT / "benchmark" / "custom_eval" / "text" / "qa_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for subset in SUBSETS:
        src = src_dir / f"{subset}.jsonl"
        records = [
            json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        out_records = [
            {
                "messages": [
                    {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                    {"role": "user", "content": r["query"]},
                ],
                "answer": r.get("answer", ""),
            }
            for r in records
        ]
        out = out_dir / f"{subset}.jsonl"
        out.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in out_records) + "\n",
            encoding="utf-8",
        )
        total += len(out_records)
        print(f"{subset}: {len(out_records)} 条")

    print(f"合计 {total} 条 → {out_dir}")
    print(f"系统提示词（基线 Prompt Engineering）：\n{BASELINE_SYSTEM_PROMPT}")


if __name__ == "__main__":
    main()
