# 四路召回 Top50 / Top100 深度消融

## 实验目的

比较 Recent、Popularity、ItemCF、Item2Vec 四路召回在每路最多返回 50 个候选和
每路最多返回 100 个候选时的覆盖能力、合并候选规模及最终排序效果。

## 公平性口径

- 数据：RetailRocket Top3000 商品、20,000 个合格 Session。
- 切分：按目标时间排序后固定为 14,000/3,000/3,000 个训练、验证、测试 Session。
- 本报告只使用验证集，未查看测试集。
- 两组使用相同标签、Point-in-Time 日快照、模型参数和随机种子 2026。
- 唯一主动变量是 `candidates_per_source`：50 对比 100。
- Item2Vec 的 `max_neighbors` 在两组中均为 100；Top100 组只是使用更多已训练近邻，
  没有改变 Item2Vec 模型。
- Top100 候选的前 50 名与 Top50 实验逐行一致。
- 所有 Point-in-Time 审计均通过，参考事件严格早于目标时间。
- 最终排序指标固定为 Recall@20；扩大候选后重新构造训练候选、困难负样本并训练排序器。

验证集共有 3,000 个标签，其中 clicks 2,623、carts 163、orders 214。

## 单路召回结果

以下 Recall 的分母都是全部 3,000 个验证标签，不按行为权重加权。

| 召回源 | Recall@20 | Recall@50 | Recall@100 | @50→@100 绝对变化 |
|---|---:|---:|---:|---:|
| Recent | 0.6663 | 0.6663 | 0.6663 | +0.0000 |
| Popularity | 0.0343 | 0.0687 | 0.1223 | +0.0537 |
| ItemCF | 0.4200 | 0.4913 | 0.5177 | +0.0263 |
| Item2Vec | 0.2113 | 0.3573 | 0.4500 | +0.0927 |

Recent 每个 Session 平均只有 2.32 个不同历史商品，最大 29 个，因此将上限从 50
改为 100 不会产生任何新增候选。Top100 下，Popularity 固定产生 100 个候选；
ItemCF 平均产生 64.68 个；Item2Vec 平均产生 98.90 个。

## 四路合并候选

| 指标 | 每路 Top50 | 每路 Top100 | 绝对变化 |
|---|---:|---:|---:|
| Candidate Recall | 0.9213 | 0.9360 | +0.0147 |
| clicks Candidate Recall | 0.9127 | 0.9291 | +0.0164 |
| carts Candidate Recall | 0.9571 | 0.9632 | +0.0061 |
| orders Candidate Recall | 1.0000 | 1.0000 | +0.0000 |
| Weighted Candidate Recall | 0.9784 | 0.9819 | +0.0035 |
| 每 Session 平均去重候选数 | 119.03 | 227.95 | +108.92 |
| 候选 CSV 行数 | 424,531 | 797,725 | +373,194 |

Top100 比 Top50 新增 44 个目标命中，其中 clicks 43 个、carts 1 个、orders 0 个。
这些新增命中在各路尾部的非互斥归因是：Popularity 6 个、ItemCF 18 个、
Item2Vec 24 个。总数大于 44 是因为同一目标可能被多路同时命中。

## 排序结果

| 排序方案 | 每路 Top50 Weighted Recall@20 | 每路 Top100 Weighted Recall@20 | 绝对变化 |
|---|---:|---:|---:|
| 启发式排序 | 0.7735 | 0.6135 | -0.1600 |
| Tree | 0.9670 | 0.9683 | +0.0013 |
| Shared-Bottom | 0.9698 | 0.9698 | +0.0000 |

Shared-Bottom 的分任务结果在两组中也完全相同：

- clicks Recall@20：0.8448
- carts Recall@20：0.9509
- orders Recall@20：1.0000

启发式排序显著下降，说明其跨召回源分数融合对候选规模敏感：新增的大量尾部候选会
挤压原来的前 20，但启发式分数没有能力稳定地区分它们。学习排序器受到的影响较小。

## 结论

1. Top100 确实提高了召回覆盖率，但 Candidate Recall 只增加 1.47 个百分点，
   Weighted Candidate Recall 只增加 0.35 个百分点。
2. 平均去重候选数从 119.03 增至 227.95，增加约 91.5%，候选计算和特征计算成本
   接近翻倍，但主排序模型没有获得 Weighted Recall@20 收益。
3. 新增覆盖主要是 clicks；orders 在 Top50 时已经达到 100%，扩大候选无法继续提升
   权重最高的 orders，因此加权收益很小。
4. Item2Vec 的 51–100 名有最高的单路增量，但这些命中与其他召回源存在重叠，
   且新增目标未被 Shared-Bottom 排入前 20。
5. 在当前结果下，应保留每路 Top50 作为默认配置。若要继续优化，更合理的方向是
   分路配额搜索，例如 Recent 使用实际历史、Popularity 20/50、ItemCF 50/100、
   Item2Vec 50/100，而不是把四路统一扩大到 100。

## 结果文件

- Top50：`results/topk_per_source_ablation/top50_validation/`
- Top100：`results/topk_per_source_ablation/top100_validation/`
- 拆分统计：`results/topk_per_source_ablation/analysis.json`
- Top100 配置：`configs/retailrocket_top100_per_source.json`

本实验仅运行单个固定随机种子，0.0013 量级的 Tree 排序增益不能解释为稳定收益；
如需据此决定生产配置，应补充多随机种子或配对 Bootstrap 置信区间。
