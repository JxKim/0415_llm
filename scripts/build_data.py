"""
数据构造脚本：种子清单 → LLM 扩写变体 → SFT messages 格式 JSONL

用法：
    python scripts/build_data.py --mode small                 # 验货：每类 10 条
    python scripts/build_data.py --mode full                  # 全量：每类 ~500 条
    python scripts/build_data.py --mode small --category 2    # 只生成某一类

API 配置（二选一）：
    1. 环境变量 DEEPSEEK_API_KEY
    2. 文件 data/.deepseek_key（纯文本，只含 key）
    可选：DEEPSEEK_MODEL（默认 deepseek-chat）

输出：data/generated/category{N}_{name}.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

SEED_FILES = {
    1: "data/seeds/seed_category1_knowledge_qa.jsonl",
    2: "data/seeds/seed_category2_diagnosis.jsonl",
    3: "data/seeds/seed_category3_scripts.jsonl",
    4: "data/seeds/seed_category4_complaints.jsonl",
}
CATEGORY_NAMES = {1: "知识问答", 2: "故障诊断", 3: "话术生成", 4: "投诉处理"}

# ---------------------------------------------------------------------------
# 每类的扩写 prompt 模板（锁死事实/骨架，只放开口语与细节多样性）
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES = {
1: """你是汽车售后客服领域的资深数据工程师。请根据给定的【种子问题】和【答案要点】，生成 {n} 条不同的训练样本。

要求：
1. 每条样本 = 车主的提问（user）+ 客服的回复（assistant），严格按此结构输出。
2. 提问要多样化：换车型（轿车/SUV/新能源）、换里程或时间、换语气（礼貌/随意/着急/怀疑）、换提问角度（直接问/带前提/追问），全部是真实车主口语，不要书面化。
3. 回复必须基于【答案要点】重组表述，**事实内容不得改变**：价格给合理区间并注明"以门店实际为准"，政策必须写明条件（如"6个月或5000公里内，先到为准"）。
4. 回复口语自然、礼貌专业，长度 50~150 字。
5. 只输出 JSON 数组，不要输出任何其他文字。

【种子问题】{seed}
【答案要点】{points}

输出格式：[{{"user": "...", "assistant": "..."}}, ...]""",

2: """你是汽车售后客服领域的资深数据工程师。下面是故障诊断的【规则】：症状、可能原因、紧急程度、标准建议、口语说法示例。请生成 {n} 条不同的训练样本。

要求：
1. 每条样本 = 车主描述症状（user）+ 客服分诊回复（assistant）。
2. user 用多样化的车主口语描述**同一症状**：换场景（高速/市区/冷天/雨天）、换情绪（紧张/平静/着急/害怕）、换说法（可参考口语说法示例但不要照抄）。
3. assistant 必须**严格按【紧急程度】和【建议】生成回复**：
   - 禁止改变紧急程度判断（高=立即停车/呼叫拖车/禁止行驶；中=尽快进店；低=可预约）
   - 禁止自由发挥修车建议（不指定具体零件型号、不承诺维修效果）
   - 结构：判断紧急程度 → 建议动作 → 到店检测项目
4. 回复长度 60~160 字，口语自然、态度负责。
5. 只输出 JSON 数组，不要输出任何其他文字。

【症状】{seed}
【可能原因】{causes}
【紧急程度】{urgency}
【标准建议】{advice}
【口语说法示例】{phrasings}

输出格式：[{{"user": "...", "assistant": "..."}}, ...]""",

3: """你是汽车售后客服话术专家。下面是【业务场景】和【话术要点】。请生成 {n} 条不同的训练样本。

要求：
1. 每条样本 = 客户的诉求（user）+ 客服的回复话术（assistant）。
2. user 是客户在该场景下的口语化诉求（如"我周末想来保养，还有位置吗？"），语气/具体时间/对象可有变化，但矛盾点与场景一致。
3. assistant 按话术骨架【{structure}】生成完整回复，**必须覆盖【话术要点】里的所有要素**（道歉/缓冲、客观事实、替代方案、促行动），口语自然、礼貌专业，长度 80~180 字。
4. 涉及价格/赔偿的口径保守：用"以门店确认为准"，不承诺具体减免金额。
5. 只输出 JSON 数组，不要输出任何其他文字。

【业务场景】{scenario}
【话术要点】{points}

输出格式：[{{"user": "...", "assistant": "..."}}, ...]""",

4: """你是汽车售后客诉处理专家。下面是【投诉场景】的客户原话和【回复要点】。请生成 {n} 条不同的训练样本。

要求：
1. 每条样本 = 客户投诉原话（user）+ 客服投诉处理回复（assistant）。
2. user 是情绪化的客户投诉：可换细节（车型/金额/次数/时间）、换情绪强度、换威胁方式（投诉/曝光/堵门/12315/厂家），保持口语化和愤怒感，但矛盾点与场景一致。
3. assistant 按骨架【{structure}】生成，**必须覆盖【回复要点】里的所有要素**：共情承认（道歉、理解、不辩解）、责任界定（免费复检/调记录、"若确认是我们的责任"）、补救方案（具体可执行但不过度承诺金额）、升级路径（400服务监督热线/店长跟进）。
4. 回复长度 80~200 字，真诚、专业。
5. 只输出 JSON 数组，不要输出任何其他文字。

【投诉场景】{scenario}
【客户原话】{customer}
【回复要点】{points}

输出格式：[{{"user": "...", "assistant": "..."}}, ...]""",
}

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _user_env_var(name: str) -> str | None:
    """Windows：从用户级环境变量（注册表 HKCU\\Environment）读取。"""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            value, _ = winreg.QueryValueEx(k, name)
            return str(value)
    except Exception:  # noqa: BLE001
        return None


def get_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        return key.strip()
    # 兜底：Windows 用户级环境变量（进程未继承时）
    user_key = _user_env_var("DEEPSEEK_API_KEY")
    if user_key:
        return user_key.strip()
    key_file = ROOT / "data" / ".deepseek_key"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    raise SystemExit(
        "[ERROR] 未找到 DeepSeek API Key：请设置环境变量 DEEPSEEK_API_KEY，"
        "或写入文件 data/.deepseek_key"
    )


def call_llm(prompt: str, max_tokens: int = 4096, temperature: float = 1.1,
             json_mode: bool = True) -> str:
    """调用 DeepSeek（OpenAI 兼容接口），失败重试 5 次。

    json_mode=True 时强制 JSON 输出（build_data 的数据生成用）；
    json_mode=False 时输出纯文本（如 DPO 改写回复）。
    """
    key = get_api_key()
    body_dict = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body_dict["response_format"] = {"type": "json_object"}
    body = json.dumps(body_dict, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    last_err = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"LLM 调用失败（5 次重试后）: {last_err}")


def parse_json_array(text: str) -> list:
    """从 LLM 输出中提取 JSON 数组（容忍 markdown 代码块/前后文字）。"""
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError(f"输出中未找到 JSON 数组: {text[:200]}")
    data = json.loads(m.group(0))
    if not isinstance(data, list):
        raise ValueError("JSON 顶层不是数组")
    return data


def validate_item(item: dict, category: int) -> dict | None:
    """基础校验：字段完整、非空、长度合理。返回清洗后的条目，非法返回 None。"""
    if not isinstance(item, dict):
        return None
    user = (item.get("user") or "").strip()
    assistant = (item.get("assistant") or "").strip()
    if not user or not assistant:
        return None
    if len(user) < 5 or len(assistant) < 20:
        return None
    if len(assistant) > 400:
        return None
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def load_seeds(category: int) -> list[dict]:
    path = ROOT / SEED_FILES[category]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_prompt(category: int, seed: dict, n: int) -> str:
    t = PROMPT_TEMPLATES[category]
    if category == 1:
        return t.format(
            n=n, seed=seed["seed"],
            points="；".join(seed["answer_points"]),
        )
    if category == 2:
        return t.format(
            n=n, seed=seed["symptom"],
            causes="、".join(seed["possible_causes"]),
            urgency=seed["urgency"],
            advice=seed["advice"],
            phrasings="；".join(seed["variant_phrasings"]),
        )
    if category == 3:
        return t.format(
            n=n, scenario=seed["scenario"],
            structure="→".join(seed["script_structure"]),
            points="；".join(seed["key_points"]),
        )
    return t.format(
        n=n, scenario=seed["complaint_type"],
        customer=seed["customer_message"],
        structure="→".join(seed["response_structure"]),
        points="；".join(seed["key_points"]),
    )


def generate_variants(category: int, seed: dict, total: int, per_call: int = 5) -> list[dict]:
    """为单个种子生成 total 条变体（分批调用，失败跳过该批）。"""
    results: list[dict] = []
    remaining = total
    while remaining > 0:
        n = min(per_call, remaining)
        prompt = build_prompt(category, seed, n)
        try:
            content = call_llm(prompt)
            items = parse_json_array(content)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] 种子生成失败（跳过该批）: {e}")
            break
        for it in items:
            v = validate_item(it, category)
            if v:
                results.append(v)
        remaining -= n
        time.sleep(0.5)  # 控制请求频率
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["small", "full"], default="small")
    ap.add_argument("--category", type=str, default=None,
                    help="类别 1|2|3|4，支持逗号分隔（如 2,3,4）；默认全部")
    ap.add_argument("--per-seed", type=int, default=None,
                    help="每个种子的变体总数（默认 small=10, full=20）")
    ap.add_argument("--per-call", type=int, default=5, help="每次 LLM 调用生成的条数")
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑：读取已有文件，只补跑种子变体不足的种子（不重写已有数据）")
    args = ap.parse_args()

    if args.mode == "small":
        per_seed = args.per_seed or 5
        seed_limit = 2  # small 模式：每类取 2 个种子 × 5 条变体 → 10 条/类
    else:
        per_seed = args.per_seed or 20
        seed_limit = None

    categories = [int(c) for c in args.category.split(",")] if args.category else [1, 2, 3, 4]
    out_dir = ROOT / "data" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"模式={args.mode} 每种子变体数={per_seed} 类别={categories} resume={args.resume}")
    for cat in categories:
        seeds = load_seeds(cat)[:seed_limit] if seed_limit else load_seeds(cat)
        print(f"--- 类别{cat} {CATEGORY_NAMES[cat]}: {len(seeds)} 个种子 ---")
        out_path = out_dir / f"category{cat}_{CATEGORY_NAMES[cat]}.jsonl"

        # 断点续跑：加载已有条目，统计每个种子的条数
        existing: list[dict] = []
        if args.resume and out_path.exists():
            existing = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if not args.resume and out_path.exists():
            out_path.unlink()  # 非续跑模式：重新生成，避免 append 冲突

        seen_user: set[str] = {it["messages"][0]["content"] for it in existing}
        have_count: dict[int, int] = {}
        for it in existing:
            idx = it.get("seed_idx", 0)
            have_count[idx] = have_count.get(idx, 0) + 1

        all_items = list(existing)
        for i, seed in enumerate(seeds, 1):
            needed = per_seed - have_count.get(i, 0)
            if args.resume and needed <= 0:
                print(f"  种子 {i}/{len(seeds)} 已有 {have_count.get(i, 0)} 条，跳过")
                continue
            total = needed if args.resume else per_seed
            print(f"  种子 {i}/{len(seeds)} (需补 {total}) ...", end=" ", flush=True)
            variants = generate_variants(cat, seed, total, args.per_call)
            batch: list[dict] = []
            for v in variants:
                u = v["messages"][0]["content"]
                if u in seen_user:
                    continue
                seen_user.add(u)
                v["seed_idx"] = i  # 标记种子来源，供按种子分组切分
                batch.append(v)
            all_items.extend(batch)
            # 边生成边追加写文件，中途失败不丢已生成部分
            with out_path.open("a", encoding="utf-8") as f:
                for it in batch:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            print(f"+{len(batch)} 条")
            time.sleep(0.3)

        print(f"  ✅ 类别{cat}: 共 {len(all_items)} 条 → {out_path}")


if __name__ == "__main__":
    main()
