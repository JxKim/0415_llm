# -*- coding: utf-8 -*-
"""核实三轮评估的配置是否都带标准答案"""
import json
from pathlib import Path

# 1. eval_runner 的 judge 配置
src = Path('scripts/eval_runner.py').read_text(encoding='utf-8')
print('=== eval_runner.py ===')
print('judge model_id 含 flash:', 'deepseek-v4-flash' in src)
print('prompt 含 {gold} 占位符:', '{gold}' in src)

# 2. 三个评估集是否都带 answer
for name, path in [
    ('qa（SFT/DPO 评估用）', 'benchmark/custom_eval/text/qa/category1_knowledge_qa.jsonl'),
    ('qa_baseline（基线评估用）', 'benchmark/custom_eval/text/qa_baseline/category1_knowledge_qa.jsonl'),
]:
    it = json.loads(Path(path).read_text(encoding='utf-8').splitlines()[0])
    has = bool(it.get('answer'))
    print(f'{name}: 字段={list(it.keys())}, answer存在={has}')

# 3. 三份评估的 judge 模型确认（从 task_config 或日志）
import re
for tag, log in [('sft_model_ref', 'output/eval_sft_ref.log'),
                 ('baseline', 'output/eval_baseline.log'),
                 ('dpo_model', 'output/eval_dpo.log')]:
    text = Path(log).read_text(encoding='utf-8', errors='ignore')
    m = re.search(r'Creating model (\S+)', text)
    judge = m.group(1) if m else '未找到'
    print(f'{tag}: 日志中 judge 模型 = {judge}')
