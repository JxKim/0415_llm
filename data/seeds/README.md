# 种子清单（Seed List）说明

> 对应方案：`微调案例实施方案.md` 第 5 节「数据构造方法」
> 场景：汽车售后 / 4S 店客服助手

## 一、什么是种子清单

**种子清单 = 人工整理的一批"代表性输入样本"，是 LLM 批量构造数据的起点模板。**

它不是训练数据本身，而是"数据的数据"——每条种子是一个真实高频场景/问题的原型，后续由 LLM 扩写成多样化变体，再经规则校验和人工抽检，最终成为训练/评估数据。

```
种子清单（人工，本目录 4 个文件）
   │  每条种子 = 真实高频场景/问题的"原型"
   ▼
LLM 扩写（每条种子生成 N 个变体）→ 多样化训练数据
   ▼
规则校验 + 人工抽检 → 最终数据集
```

## 二、种子清单的三个作用

1. **控制数据分布**：每个类别下覆盖哪些子主题、各占多少，由人工决定，防止 LLM 生成时全部偏到某一个方向
2. **保证质量**：种子本身是真实的、正确的，LLM 只做"扩写变体"，防止 LLM 自由发挥编出错误内容（尤其故障诊断，这是安全红线）
3. **可复现**：种子清单是纯人工产物，任何人都能看懂数据从哪来、为什么这么设计

## 三、4 类种子的形态差异

| 文件 | 类别 | 种子形态 | 字段 |
|---|---|---|---|
| `seed_category1_knowledge_qa.jsonl` | 1 售后知识问答 | 车主高频问题 + 答案要点 | subtopic / seed / answer_points |
| `seed_category2_diagnosis.jsonl` | 2 故障诊断与安全分级 | **知识规则表**（症状→原因→紧急程度→建议） | symptom / possible_causes / urgency / advice / variant_phrasings |
| `seed_category3_scripts.jsonl` | 3 服务话术生成 | 为难场景枚举 + 话术骨架要点 | scenario / script_structure / key_points |
| `seed_category4_complaints.jsonl` | 4 投诉处理与安抚 | 投诉类型 + 客户原话 + 回复要点 | complaint_type / customer_message / response_structure / key_points |

各类种子数量：每类 25 条，共 100 条。

## 四、构造时的使用方式（后续脚本）

1. **类别 1**：每条种子扩写 N 个口语变体（换车型、换里程、换语气），答案按 answer_points 重组表述
2. **类别 2**：每条规则用 variant_phrasings 做模板扩写更多口语症状描述，回复严格按 possible_causes + urgency + advice 生成，**禁止 LLM 自由发挥修车建议**
3. **类别 3**：按 script_structure 骨架 + key_points 扩写成完整话术
4. **类别 4**：按 response_structure 骨架 + key_points 扩写成完整投诉处理回复；同时按"反要点"刻意生成 rejected 版本供 DPO 使用

## 五、质量红线（构造与校验时强制）

- 故障诊断：紧急程度分级保守（高紧急度绝不建议继续行驶）
- 投诉处理：不推诿、不无条件承诺赔偿金额（用"若确认是我们的责任"表述）
- 价格/政策：给区间并注明"以门店实际为准"
- 话术生成：只拒绝不给方案 = 不合格，必须"给出路"
