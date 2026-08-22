# -*- coding: utf-8 -*-
"""诊断2：检查类别2原始生成数据的紧急程度标注"""
import json
from collections import Counter
from pathlib import Path

items = [json.loads(l) for l in Path('data/generated/category2_故障诊断.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
print(f'类别2 原始生成: {len(items)} 条')

hi = sum(1 for it in items if '高紧急' in it['messages'][1]['content'] or '禁止' in it['messages'][1]['content'] or '立即停车' in it['messages'][1]['content'] or '绝对不能' in it['messages'][1]['content'])
mid = sum(1 for it in items if ('中紧急' in it['messages'][1]['content'] or '尽快' in it['messages'][1]['content']) and '高紧急' not in it['messages'][1]['content'] and '禁止' not in it['messages'][1]['content'])
low = sum(1 for it in items if '低紧急' in it['messages'][1]['content'] or '预约' in it['messages'][1]['content'])
nolevel = len(items) - hi - mid - low
print(f'含高紧急/禁止/立即停车: {hi} | 中(尽快): {mid} | 低(预约): {low} | 无明确分级: {nolevel}')

print()
print('=== 无明确分级表述的样本（前 6 条）===')
cnt = 0
for it in items:
    a = it['messages'][1]['content']
    if not ('紧急' in a or '禁止' in a or '预约' in a):
        print('问:', it['messages'][0]['content'][:70])
        print('答:', a[:200])
        print('---')
        cnt += 1
        if cnt >= 6:
            break

print()
print('=== 刹车软种子(seed 1)的样本回复示例 ===')
cnt = 0
for it in items:
    if it.get('seed_idx') == 1:
        print('问:', it['messages'][0]['content'][:70])
        print('答:', it['messages'][1]['content'][:220])
        print('---')
        cnt += 1
        if cnt >= 4:
            break
