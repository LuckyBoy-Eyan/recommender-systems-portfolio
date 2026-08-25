# Point-in-Time Warm-up 窗口消融

## 实验问题

检验在完全固定正式标签Session的前提下，给全局Popularity、ItemCF、Item2Vec和
商品统计增加目标期之前的历史预热，能否缓解最早快照数据不足并提升候选与排序效果。

## 固定协议

- 原始数据永久保存在 `data/retailrocket_raw/events.csv`；
- 原始文件SHA-256：
  `3745aa83238b1e6d44d8fda209807899f420084398f94ddf745f3cbcfecbf9e7`；
- 扩大预处理池：Top3000、30,000个合格Session、145,097条事件；
- 正式标签起点：`2015-05-17 00:00:00 UTC`；
- 从起点之后按目标时间固定选择20,000个正式Session；
- 固定切分：14,000训练、3,000验证、3,000测试；
- 四组验证标签SHA-256：
  `a6a834343e1d4a28c7882874d6c344fa73df5e942607abbf891a87d70f857749`；
- Warm-up Session不产生标签，只进入Point-in-Time全局参考池；
- 0/3/7/14天组使用完全相同的标签、随机种子、候选配额、Item2Vec参数和排序器；
- 本轮只比较验证集，不根据测试集选择Warm-up长度。

Warm-up池均由完整落在对应窗口内的Session组成：

| 窗口 | Warm-up Session | Warm-up事件 |
|---:|---:|---:|
| 0天 | 0 | 0 |
| 3天 | 1,120 | 5,365 |
| 7天 | 2,382 | 11,252 |
| 14天 | 4,743 | 22,437 |

## 验证结果

| Warm-up | Candidate Recall | Weighted Candidate Recall | Item2Vec R@20 | ItemCF R@20 | Tree Weighted R@20 | Shared-Bottom Weighted R@20 | 总耗时 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0天 | 0.9250 | 0.98208 | 0.2010 | 0.4207 | 0.96506 | 0.97202 | 236.5秒 |
| 3天 | 0.9230 | 0.98407 | 0.1990 | 0.4223 | 0.96668 | **0.97450** | 259.6秒 |
| 7天 | 0.9270 | 0.98452 | **0.2073** | 0.4270 | 0.96923 | 0.96972 | 273.2秒 |
| 14天 | **0.9287** | **0.98471** | **0.2073** | **0.4377** | **0.96935** | 0.97247 | 316.7秒 |

Recent完全不依赖全局Warm-up，四组Recall@20均为`0.6687`，验证了正式Session
局部历史没有变化。

## 相对0天的逐Session候选变化

| Warm-up | 新增命中 | 丢失命中 | 净变化 | 点击净变化 | 加购净变化 | 购买净变化 |
|---:|---:|---:|---:|---:|---:|---:|
| 3天 | 20 | 26 | -6 | -7 | +1 | 0 |
| 7天 | 23 | 17 | +6 | +5 | +1 | 0 |
| 14天 | 29 | 18 | +11 | +10 | +1 | 0 |

Item2Vec Top20相对0天的命中集合也会替换，并非只新增不丢失：

| Warm-up | Item2Vec新增命中 | Item2Vec丢失命中 | 净变化 |
|---:|---:|---:|---:|
| 3天 | 76 | 82 | -6 |
| 7天 | 103 | 84 | +19 |
| 14天 | 100 | 81 | +19 |

## 结论

1. Warm-up确实解决了最早快照历史不足：首个训练快照参考事件从0天组的9条，
   增加到3/7/14天组的5,374/11,261/22,446条。
2. 7天和14天均使Item2Vec Top20净增加19个命中；14天还显著提高ItemCF，
   最终合并候选净增加11个目标。
3. 候选覆盖提升没有单调转化为Shared-Bottom Top20提升。按预先设定的主排序器
   验证Weighted Recall@20选择，3天组最高，但相对0天绝对提升仅`0.00248`。
4. 3天组的加权提升主要来自多命中1个加购目标，以及少量点击排序变化；当前验证
   只有133个加购标签，一个样本即可改变约`0.00226`的加权指标，因此证据较弱。
5. 14天使总耗时从236.5秒增加到316.7秒，约增加33.9%，需要同时考虑计算成本。
6. 当前结果只使用一个固定种子，不能宣称3天稳定优于其他窗口。下一步应先对0/3/7/14
   做多随机种子验证；如果必须按当前单次主指标锁定方案，则选择3天，随后只评估一次测试集。

## 复现

```bash
python scripts/run_pipeline.py \
  --config configs/retailrocket_warmup_ablation.json \
  --warmup-days 0 \
  --output outputs/warmup_ablation/day0_validation

python scripts/run_pipeline.py \
  --config configs/retailrocket_warmup_ablation.json \
  --warmup-days 3 \
  --output outputs/warmup_ablation/day3_validation

python scripts/run_pipeline.py \
  --config configs/retailrocket_warmup_ablation.json \
  --warmup-days 7 \
  --output outputs/warmup_ablation/day7_validation

python scripts/run_pipeline.py \
  --config configs/retailrocket_warmup_ablation.json \
  --warmup-days 14 \
  --output outputs/warmup_ablation/day14_validation
```
