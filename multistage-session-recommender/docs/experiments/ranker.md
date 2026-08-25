# Tree与Shared-Bottom排序比较

## 公平比较

两个排序器使用完全相同的：

- Point-in-Time 训练与评估快照；
- Recent、Popularity、ItemCF、Item2Vec 候选；
- 排序特征与列顺序；
- 每个 Session 的 Hard Negative；
- clicks、carts、orders 标签；
- 数据切分和评估代码。

Tree 为三个目标分别训练独立模型。Shared-Bottom 使用 `input → 64 → 32` 的共享底层，
再接三个独立任务塔，共有 3,843 个可训练参数。训练 10 轮，batch size 为 1024，
Adam 学习率为 0.001，权重衰减为 0.0001。

## 标签与Mask

每个 Session 只留出最后一个事件，因此一行候选只对其 `target_type` 有监督。例如购买
目标对应：

```text
labels = [0, 0, order_label]
masks  = [0, 0, 1]
```

未观测的点击和加购标签不作为负例。每个任务只在 `mask=1` 的行上计算
`BCEWithLogitsLoss`，三个有效任务损失等权平均；任务内 `pos_weight` 使用负例数除以
正例数，缓解类别不平衡。

## 正式结果

| 数据集 | Tree Weighted Recall@20 | Shared-Bottom | 差值 |
|---|---:|---:|---:|
| 验证 | 0.9670 | **0.9698** | +0.00273 |
| 测试 | 0.9665 | **0.9668** | +0.00030 |

测试分任务：

| 模型 | clicks | carts | orders |
|---|---:|---:|---:|
| Tree | 0.8514 | 0.9563 | 0.9907 |
| Shared-Bottom | 0.8544 | 0.9563 | 0.9907 |

验证集按预定指标选择 Shared-Bottom 是合理的，但测试差值只有 `0.00030`。当前只有一个
正式随机种子，不能声称多任务共享结构带来显著或稳定收益。可信结论是：两者在当前任务
上基本持平，Shared-Bottom 仅有轻微数值优势。

若要继续比较 Independent MLP、MMOE 或 PLE，应在新的验证方案上复用同一候选缓存和
训练样本，并报告多随机种子、参数量、耗时、任务级指标及配对差值。
