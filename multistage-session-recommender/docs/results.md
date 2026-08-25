# 实验报告

## 实验问题

在不把留出的目标事件用于任何全局统计的前提下，分目标排序模型是否能相比多路召回
RRF 排名融合基线带来提升？

## 实验协议

- 按目标事件时间将完整 Session 划分为 70% 训练集、15% 验证集和 15% 测试集
- 每个 Session 留出最后一个事件作为预测标签
- 每个样本使用严格早于目标时间的时间快照构建全局召回图和物品特征
- 召回出来但没有命中真实目标的候选商品作为 Hard Negative
- 固定随机种子：2026

## 实验结果

当前正式数据口径为Top3000、20,000个Session、96,516条行为和2,802个实际出现
商品。训练、验证、测试Session为14,000/3,000/3,000。

### 验证集

| 指标 | 数值 |
|---|---:|
| 整体Candidate Recall | 0.9213 |
| Weighted Candidate Recall | 0.9784 |
| RRF Weighted Recall@20 | 0.7735 |
| Shared-Bottom Weighted Recall@20 | **0.9698** |

### 锁定测试集

| 指标 | 数值 |
|---|---:|
| 整体Candidate Recall | 0.9310 |
| 点击/加购/购买 Candidate Recall | 0.9234 / 0.9688 / 0.9954 |
| Weighted Candidate Recall | 0.9802 |
| Recent / ItemCF / Item2Vec / Popularity Recall@20 | 0.6737 / 0.4380 / 0.2197 / 0.0357 |
| RRF Weighted Recall@20 | 0.7770 |
| Shared-Bottom Weighted Recall@20 | **0.9668** |

Shared-Bottom相对RRF绝对提升`0.1898`、相对提升约`24.43%`。

全部日级快照通过`max_reference_ts < min_target_ts`审计。端到端训练、验证和测试
总耗时约304.55秒。

## 结果解释

ItemCF 提供主要的协同相似召回覆盖，Item2Vec 补充基于序列上下文的召回路径。
Co-visitation 因测试集没有独占目标命中且与 ItemCF 高度冗余，已在消融后删除。分目标
排序模型能够对转化率等特征赋予不同权重，这一点很重要，因为 OTTO 风格评估会给
orders 更高权重。

## 生产环境注意事项

- 按事件时间窗口离线物化召回图。
- 使用满足 point-in-time 正确性的特征表。
- 在关注排序指标之前，先监控候选集合 Recall。
- 保持离线事件定义和线上日志语义一致。
- 在大规模真实数据上，可进一步比较LambdaRank等直接优化排序目标的方案。
