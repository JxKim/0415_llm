# -*- coding: utf-8 -*-
"""诊断：检查 SFT 训练数据中类别2（故障诊断）的紧急程度标注质量"""
import json
from pathlib import Path

items = [json.loads(l) for l in Path('data/sft_train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

cat2 = []
for it in items:
    a = it['messages'][1]['content']
    if any(k in a for k in ['紧急', '拖车', '制动', '故障灯', '冷却液', '发动机', '轮胎']):
        cat2.append(it)
print(f'疑似类别2样本数: {len(cat2)}')

hi = sum(1 for it in cat2 if '高紧急' in it['messages'][1]['content'] or '禁止' in it['messages'][1]['content'] or '立即停车' in it['messages'][1]['content'])
mid = sum(1 for it in cat2 if '中紧急' in it['messages'][1]['content'] and '高紧急' not in it['messages'][1]['content'])
low = sum(1 for it in cat2 if '低紧急' in it['messages'][1]['content'])
nolevel = len(cat2) - hi - mid - low
print(f'含高紧急/禁止/立即停车: {hi} | 含中紧急: {mid} | 含低紧急: {low} | 无明确分级: {nolevel}')

print()
print('=== 无明确分级表述的样本示例（前 5 条）===')
cnt = 0
for it in cat2:
    a = it['messages'][1]['content']
    if not ('高紧急' in a or '中紧急' in a or '低紧急' in a):
        print('问:', it['messages'][0]['content'][:70])
        print('答:', a[:160])
        print('---')
        cnt += 1
        if cnt >= 5:
            break

print()
print('=== 刹车/制动类样本（user含刹车或制动）的回复分级统计 ===')
brake = [it for it in items if '刹车' in it['messages'][0]['content'] or '制动' in it['messages'][0]['content']]
for tag, cond in [('高', lambda a: '高紧急' in a or '禁止' in a or '立即停车' in a),
                  ('中', lambda a: '中紧急' in a),
                  ('低', lambda a: '低紧急' in a),
                  ('无分级', lambda a: not ('紧急' in a))]:
    n = sum(1 for it in brake if cond(it['messages'][1]['content']))
    print(f'  刹车类样本({len(brake)}条): 回复含[{tag}]分级表述 {n} 条')
