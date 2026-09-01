# RetailRocket 早期 Top3000 实验（历史归档）

> 本文记录早期 Top3000、四路召回和 Shared-Bottom 实验，不代表当前正式系统。当前结果请阅读[完整实验报告](complete-experiment-report.md)与[结果摘要](results.md)。

## 数据处理

RetailRocket 事件映射如下：

- `view -> clicks`
- `addtocart -> carts`
- `transaction -> orders`

处理顺序：

1. 按访客和时间排序；
2. 相邻行为间隔超过 30 分钟时开启新 Session；
3. 按全量事件频次保留 Top3000 商品；
4. 删除商品过滤后少于三次行为的 Session；
5. 删除最大时间戳并列、无法唯一确定目标事件的 Session；
6. 按原始 Session 开始时间选择最早的合格 Session，删除后从后续递补至 20,000 个。

本次共排除 139 个目标时间歧义 Session，最终数据包含：

| 项目 | 数量 |
|---|---:|
| Session | 20,000 |
| 实际出现商品 | 2,802 |
| 行为 | 96,516 |
| 点击 | 87,293 |
| 加购 | 6,547 |
| 购买 | 2,676 |

## 实验协议

- 按 Session 最后一个事件的时间切成 70%/15%/15%，即 14,000/3,000/3,000；
- 每个 Session 留出最后一个事件作为预测目标；
- 日级快照中的全局事件严格满足 `event_ts < snapshot_ts <= target_ts`；
- Session 局部历史只使用目标事件之前的行为；
- Recent、Popularity、ItemCF、Item2Vec 生成候选；
- 使用Shared-Bottom完成点击、加购和购买多目标排序；
- 随机种子为 2026；
- 根据验证集选择主排序器，配置锁定后执行一次测试评估。

当前 Top3000 目录由全量事件频次确定，因此目录选择属于 transductive 设定。严格
Point-in-Time 的声明只覆盖动态召回图、Item2Vec、商品统计和排序特征。若要求目录也
完全因果，需要仅用训练时间窗口选择商品并重新生成数据与结果。

## 正式结果

| 数据集 | Candidate Recall | Weighted Candidate Recall | RRF | Shared-Bottom |
|---|---:|---:|---:|---:|
| 验证 | 0.9213 | 0.9784 | 0.7735 | **0.9698** |
| 测试 | 0.9310 | 0.9802 | 0.7770 | **0.9668** |

测试集分行为结果：

| 指标 | clicks | carts | orders |
|---|---:|---:|---:|
| Candidate Recall | 0.9234 | 0.9688 | 0.9954 |
| RRF Recall@20 | 0.7504 | 0.7750 | 0.7824 |
| Shared-Bottom Recall@20 | 0.8544 | 0.9563 | 0.9907 |

测试集各召回源独立 Recall@20：

| Recent | Popularity | ItemCF | Item2Vec |
|---:|---:|---:|---:|
| 0.6737 | 0.0357 | 0.4380 | 0.2197 |

Shared-Bottom相对RRF的绝对提升为`0.1898`，相对提升约`24.43%`。

## 指标定义

整体 Candidate Recall 不区分行为：

\[
\mathrm{CandidateRecall}
=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(y_i\in C_i)
\]

Weighted Candidate Recall 与 Weighted Recall@20 都先分别计算 clicks、carts、orders，
再按 `0.1/0.3/0.6` 加权。前者检查目标是否进入完整候选集，后者检查目标是否进入排序后
前 20。二者只有在相同任务权重下才能直接比较。

## Point-in-Time审计

验证和测试各物化 9 个日级快照，所有快照均通过因果性检查。最终产物保存：

- 配置与数据 SHA256 指纹；
- 候选和每路召回明细；
- 快照时间审计；
- 可加载排序器；
- 运行环境与分阶段耗时；
- 验证和测试指标。

最终端到端运行耗时约 304.55 秒。

## 局限

- 结果来自 RetailRocket 子集，不是 OTTO 榜单成绩；
- 高频商品目录不适合评估完整长尾覆盖；
- Leave-Last-Event-Out 每个 Session 只预测一个目标；
- 日级快照严格因果，但牺牲当天特征新鲜度；
- 测试回放允许较晚样本使用此前已经发生的测试事件，模拟滚动更新；
- 目前只有一个正式随机种子，模型间小差异不能解释为稳定收益；
- 测试集已经打开，后续参数选择只能使用新的验证方案或重新划分数据。
