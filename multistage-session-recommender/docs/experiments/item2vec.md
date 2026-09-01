# Item2Vec实现与当前结果

## 实现

Item2Vec 使用 PyTorch 实现 Skip-Gram Negative Sampling：

- 每个 Session 按时间排序，形成商品 Token 序列；
- 窗口内中心商品和上下文商品构成正样本；
- 负商品按 `count(aid)^0.75` 分布采样；
- 正样本最大化 `log sigmoid(v_c · v_o)`；
- 负样本最大化 `log sigmoid(-v_c · v_n)`；
- 输入向量 L2 归一化后按余弦相似度生成近邻；
- 当前商品规模可直接计算近邻，工业规模应切换 ANN。

当前配置：

```text
dimensions=32
window=2
negative_samples=5
epochs=8
batch_size=1024
learning_rate=0.01
min_count=1
max_neighbors=100
action_weights={clicks: 1, carts: 3, orders: 6}
exclude_positive_contexts=false
```

行为权重应用于正样本对，中心行为与上下文行为权重取几何平均，并归一化到均值 1。

## Point-in-Time

每个日级快照只使用：

```text
event_ts < snapshot_ts <= target_ts
```

目标当天和未来事件不会进入 Item2Vec 训练序列。相同快照固定初始化、数据顺序、
负采样随机源和 CPU 线程数，以保证可复现。

## 当前正式结果

| 数据集 | Item2Vec Recall@20 | 合并 Candidate Recall |
|---|---:|---:|
| 验证 | 0.2113 | 0.9213 |
| 测试 | 0.2197 | 0.9310 |

这些数字说明 Item2Vec 提供了一路可用候选，但不能单独证明它对合并候选或最终排序的
增量贡献。当前数据口径尚未完成 Item2Vec、SVD 与关闭向量召回的多随机种子公平消融，
因此简历和面试中不应声称 Item2Vec 稳定优于其他向量召回方案。

若继续研究，应只在新的验证方案上比较：

- 单路 Recall@20；
- 合并 Candidate Recall 与独占命中；
- 最终 Weighted Recall@20；
- 多随机种子的均值和方差；
- 快照训练时间、索引时间与内存开销。
