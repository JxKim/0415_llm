"""
评估脚本：evalscope + LLM-as-a-Judge（DeepSeek 裁判）

用法：
    python scripts/eval_runner.py --model Qwen3-0.6B-finetuned --api-url http://127.0.0.1:8000/v1
    python scripts/eval_runner.py --model Qwen3-0.6B --api-url http://127.0.0.1:8000/v1 --tag baseline

说明：
- 评估集：benchmark/custom_eval/text/qa/ 下 4 个 subset（category1~4）
- 裁判：DeepSeek API（key 读取：环境变量 → 注册表用户级 → data/.deepseek_key）
- 输出：evalscope 结果目录（默认 outputs/evalscope/{tag}）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

JUDGE_PROMPT = """你是一个汽车售后客服回复质量评测裁判。不要输出分析过程。
请判断模型回复的效果。最终输出一个分数。
最终只输出一个分数，格式必须为 [[0到1之间的小数]]，例如 [[0.8]]

评分标准：
1、格式规范：回复结构完整、要素齐全（如故障诊断需含紧急程度判断与建议、投诉处理需含共情/补救/升级路径），权重：0.3
2、信息准确：**对照【标准答案】**判断政策/价格/安全判断是否正确（涉及行车安全的分级判断错误直接给低分），权重：0.4
3、语气与共情：礼貌、专业、不推诿，权重：0.2
4、方案可执行：给出明确后续动作（到店检测/预约/拖车/留联系方式等），权重：0.1
最终将结果限制到0-1的范围

[题目]
{question}

[标准答案]
{gold}

[模型输出]
{pred}

输出一个最终的0-1的分数"""

SUBSETS = ["category1_knowledge_qa", "category2_diagnosis", "category3_scripts", "category4_complaints"]


def get_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            v, _ = winreg.QueryValueEx(k, "DEEPSEEK_API_KEY")
            if v:
                return str(v).strip()
    except Exception:  # noqa: BLE001
        pass
    key_file = ROOT / "data" / ".deepseek_key"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    raise SystemExit("[ERROR] 未找到 DeepSeek API Key（环境变量/注册表/data/.deepseek_key）")


def build_judge_args() -> dict:
    return {
        "model_id": "deepseek-v4-flash",
        "api_url": "https://api.deepseek.com",
        "api_key": get_api_key(),
        "generation_config": {"temperature": 0.0, "max_tokens": 1024},
        "score_type": "numeric",
        "score_pattern": "\\[\\[(\\d+(?:\\.\\d+)?)\\]\\]",
        "prompt_template": JUDGE_PROMPT,
    }


def main() -> None:
    # 移除代理环境变量：openai 库（evalscope 底层）走系统代理连 DeepSeek 会持续连接重试
    # 本机评估只需 localhost vLLM + 直连 DeepSeek，均不需要代理
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(var, None)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型标识（--model 参数，用于输出标识）")
    ap.add_argument("--api-url", default=None, help="vLLM API 地址（不传则本地加载模型评估）")
    ap.add_argument("--local-model", action="store_true",
                    help="本地模型模式：--model 传本地路径，不通过 vLLM（evalscope 用 transformers 加载）")
    ap.add_argument("--model-dtype", default="bfloat16", help="本地加载 dtype")
    ap.add_argument("--tag", default=None, help="结果目录标识（默认取模型名）")
    ap.add_argument("--local-path", default="benchmark/custom_eval/text/qa",
                    help="评估集目录（相对项目根）")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    tag = args.tag or args.model.replace("/", "_")
    dataset_args = {
        "general_qa": {
            "local_path": str(ROOT / args.local_path),
            "subset_list": SUBSETS,
            "prompt_template": "{query}",
        }
    }

    evalscope_cmd = str(ROOT / ".venv" / "Scripts" / "evalscope.exe")
    if not Path(evalscope_cmd).exists():
        # 云上/Linux：探测常见路径（screen 环境的 PATH 可能不含 miniconda）
        for cand in ("/root/miniconda3/bin/evalscope", "/opt/conda/bin/evalscope"):
            if Path(cand).exists():
                evalscope_cmd = cand
                break
        else:
            evalscope_cmd = "evalscope"
    cmd = [
        evalscope_cmd,
        "eval",
        "--model", args.model,
    ]
    if args.local_model:
        # 本地模型模式：不传 --api-url，用 --model-args 指定加载参数
        cmd += [
            "--model-args", json.dumps({"dtype": args.model_dtype, "device_map": "auto"}),
        ]
    else:
        cmd += ["--api-url", args.api_url]
    gen_config = {"temperature": 0.0, "max_tokens": args.max_tokens}
    if args.local_model:
        # transformers generate 不接受 temperature=0（vLLM 接受），本地模式用 do_sample=False
        gen_config = {"do_sample": False, "max_tokens": args.max_tokens}
    cmd += [
        "--datasets", "general_qa",
        "--dataset-args", json.dumps(dataset_args, ensure_ascii=False),
        "--judge-strategy", "llm",
        "--judge-model-args", json.dumps(build_judge_args(), ensure_ascii=False),
        "--generation-config", json.dumps(gen_config),
        "--eval-batch-size", str(args.batch),
        "--work-dir", str(ROOT / "outputs" / "evalscope" / tag),
    ]
    print(">>> " + " ".join(cmd[:8]) + " ...", flush=True)
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
