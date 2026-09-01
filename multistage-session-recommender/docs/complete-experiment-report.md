# RetailRocket 多阶段 Session 推荐系统：完整实验过程与最终结果

> 文档状态：初稿，供项目审查。
> 项目目标：在约 23.5 万件商品中进行全量候选召回，经过多任务精排后，为每个 Session 输出一个统一的 Top-K 商品列表。
> 最终主模型：七路召回 + PLE 多任务精排。
> 独立强基线：相同七路候选 + RRF 排序。
> 时间协议：训练、验证、测试严格按时间切分；测试集在模型、参数和融合权重冻结后只评估一次。

---

## 1. 任务定义

给定用户当前 Session 中目标时刻以前的行为序列：

```text
[(item_1, action_1, time_1), ..., (item_t, action_t, time_t)]
```

模型需要从全量商品目录中预测用户下一次交互的商品 `target_item`。下一次行为可能是点击、加购或购买，但线上推断时其行为类型不可知，因此最终不能分别返回三份 Top20，而必须返回一份统一的 Top-K 商品列表。

系统分为两阶段：

1. **召回阶段**：七路召回从全量目录产生数百至约一千个去重候选。
2. **精排阶段**：PLE 对每个 `(Session, candidate_item)` 输出点击、加购、购买三个任务分数，再融合为一个最终分数并生成统一列表。

RRF 的角色始终是独立排序基线。它不负责截断 PLE 的候选集合，也不和 PLE 分数混合。

---

## 2. 原始数据集

使用 RetailRocket 电商行为数据，包括事件、商品属性和类目树。

### 2.1 事件规模

| 项目 | 数量 |
|---|---:|
| 原始事件 | 2,756,101 |
| 删除 460 条完全重复事件后 | 2,755,641 |
| 用户数 | 1,407,580 |
| 事件商品数 | 235,061 |
| 点击 | 2,664,218 |
| 加购 | 68,966 |
| 购买 | 22,457 |

数据具有明显行为不平衡：点击占绝大多数，购买极稀疏。因此训练时既需要任务级正样本权重，也需要在最终业务指标中区分点击、加购和购买价值。

### 2.2 商品属性与类目树

| 项目 | 数量 |
|---|---:|
| 商品属性原始记录 | 20,275,902 |
| 属性种类 | 1,104 |
| 有类目状态记录的商品 | 417,053 |
| 有可用状态记录的商品 | 417,053 |
| 类目数 | 1,699 |
| 根类目数 | 55 |
| 类目树最大深度 | 5 |

重点使用的动态属性包括：

- `categoryid`：商品在对应时间点所属的类目；
- `available`：商品在对应时间点的可用状态；
- 类目路径：叶子类目、父类目、根类目、深度；
- `first_seen_ts`：商品第一次出现在数据中的时间。

这些属性均按目标时刻构造 Point-in-Time 快照。例如，预测 7 月 20 日的行为时，只允许使用 7 月 20 日以前已经出现的类目或可用状态，不能使用后来更新的属性。

---

## 3. Session 构造、清洗与时间切分

### 3.1 Session 切分

- 按用户聚合事件；
- 相邻事件时间差超过 30 分钟时切分新 Session；
- 同一 Session 内按 `timestamp + event_order` 稳定排序；
- 完全重复事件删除，其余原始事件保留。

### 3.2 哪些 Session 参与监督训练

- 少于 3 个事件的 Session 无法稳定形成“至少两步历史 + 下一步目标”，因此不作为监督样本；
- 最后时间戳存在多个并列事件、无法唯一确定下一目标的 Session，不作为监督样本；
- 这些 Session **没有从历史数据中删除**，仍用于统计 Popular、共现、转移和 Item2Vec 等召回信息；
- 它们只是没有作为带标签的训练/验证/测试目标。

统计结果：

| 项目 | 数量 |
|---|---:|
| 全部 Session | 1,761,675 |
| 可作为标签目标的 Session | 171,326 |
| 仅作为历史参考的 Session | 1,590,349 |
| 过短 Session | 1,589,497 |
| 最后时间戳并列 | 690 |
| 同时过短且并列 | 162 |

### 3.3 Session 长度分布

| 统计量 | 长度 |
|---|---:|
| 平均值 | 5.53 |
| P50 | 4 |
| P75 | 6 |
| P90 | 9 |
| P95 | 13 |
| P99 | 30 |
| 最大值 | 417 |

因此最终使用 `max_history=30`：它覆盖约 99% 的典型 Session，同时避免极长 Session 带来的计算和噪声。

### 3.4 时间切分

按目标事件时间排序后进行 70%/15%/15% 切分：

| 划分 | 原始目标 Session |
|---|---:|
| 训练 | 119,928 |
| 验证 | 25,698 |
| 测试 | 25,700 |

验证开始前冻结验证所需的召回索引；测试阶段使用训练与验证结束后冻结的最终索引。测试标签在模型、候选策略、精排结构及融合权重全部确定后才读取一次。

### 3.5 滑动前缀训练样本

训练集不只使用每个 Session 的最后一个前缀，而是对一个 Session 构造多个序列前缀：

```text
[A, B]       -> C
[A, B, C]    -> D
[A, B, C, D] -> E
```

参数：

- `min_history=2`；
- `max_history=30`；
- 每个 Session 最多 50 个训练样本；
- 时间并列、目标不唯一的前缀跳过。

最终：

| 项目 | 数量 |
|---|---:|
| 候选前缀样本 | 431,895 |
| 最终训练样本 | 414,481 |
| 跳过并列目标 | 1,701 |
| 触发每 Session 50 样本上限 | 458 |

---

## 4. 因果约束与 Point-in-Time 规则

所有召回和特征都遵守：

```text
feature_timestamp < target_timestamp
```

具体包括：

1. Popular 只统计目标时刻以前的事件；
2. ItemCF 只使用冻结时点以前的 Session 共现；
3. Transition 只统计历史中从早到晚的有向转移；
4. Item2Vec 只使用冻结时点以前的序列训练；
5. 商品类目、可用状态只取目标时刻以前最近一次状态；
6. 双塔的 Batch 内负样本使用因果掩码，避免把当时尚不可见的目标当成合法负样本；
7. 构造困难负样本时屏蔽当前 Session 在目标时刻及之后真实发生的商品；
8. 最终测试不注入漏召回的真实目标商品。

---

## 5. 七路召回总览

最终七路为：

1. Recent；
2. Hybrid Popular；
3. ItemCF；
4. Item2Vec；
5. Category；
6. Transition；
7. Two-Tower。

生产候选预算：

| 路线 | 单 Session 最大候选数 |
|---|---:|
| Recent | 30 |
| Hybrid Popular | 100 |
| ItemCF | 200 |
| Item2Vec | 250 |
| Category | 150 |
| Transition | 150 |
| Two-Tower | 300 |
| 理论相加 | 1,180 |

七路结果按 `(session, aid)` 去重。由于不同路线会召回相同商品，最终平均候选数明显小于 1,180。

---

## 6. Recent 召回

### 6.1 处理逻辑

从当前 Session 可见历史末尾向前读取，并按商品 ID 去重：

```python
recent = list(dict.fromkeys(history_aids[::-1]))
```

因此它不是“直接取最近 30 条事件”，而是“最近 30 个不同商品”。同一商品连续点击多次只保留最近一次。

### 6.2 参数与评分

- 最大召回数：30；
- 排名分数：`score = 1 / rank`；
- 第 1 个商品得分 1，第 2 个为 1/2，以此类推；
- 只使用当前 Session 自己已经出现过的商品；
- 对新目标商品的召回能力为 0，但对重复交互、加购和购买极强。

统一 Top50 诊断中：

- HitRate@50：0.52529；
- 独占命中：1,489；
- 去掉 Recent 后，联合 HitRate 下降 5.79 个百分点；
- Order HitRate@50：0.98383。

Recent 无需复杂训练，它承担的是“用户继续与刚刚浏览的商品交互”的强规则先验。

---

## 7. Hybrid Popular 召回

### 7.1 为什么不只用全局热度

纯全局 Popular 容易被长期高频点击商品垄断，覆盖窄、个性化弱。因此最终混合以下信息：

- 点击热榜；
- 加购热榜；
- 购买热榜；
- 时间趋势热度；
- 上架/首次出现较新的商品；
- 当前 Session 的行为阶段。

### 7.2 行为权重与趋势公式

事件行为权重：

```text
click = 1
cart  = 3
order = 6
```

趋势分数：

```text
trend_score = action_weight × [exp(-age_days) + 0.3 × exp(-age_days / 7)]
```

新品池定义为冻结时点以前 30 天内首次出现的商品，新品得分：

```text
new_score = action_weight × exp(-event_age_days / 7) / sqrt(item_age_days + 1)
```

### 7.3 按 Session 阶段分配配额

若历史中出现购买，则视为 `order` 阶段；否则出现加购则为 `cart` 阶段；否则为 `click` 阶段。

| 阶段 | 点击榜 | 加购榜 | 购买榜 | 趋势池 | 新品池 | 总预算 |
|---|---:|---:|---:|---:|---:|---:|
| click | 36 | 18 | 6 | 20 | 20 | 100 |
| cart | 12 | 24 | 24 | 20 | 20 | 100 |
| order | 6 | 12 | 42 | 20 | 20 | 100 |

各池按顺序去重，不足部分使用全局行为加权热榜补齐。

Popular 单路 HitRate 较低，但成本很低，且可作为冷启动和候选不足时的兜底。它在联合候选中仍提供少量独占命中，因此保留 100 个预算。

---

## 8. ItemCF 召回

### 8.1 相似度构建

先在每个历史 Session 内对商品去重。长度为 `L` 的 Session 对任意共现商品对贡献：

```text
co_weight = 1 / log2(2 + L)
```

这样降低超长 Session 中偶然共现的影响。随后进行余弦形式归一化：

```text
sim(i, j) = co_count(i, j) / sqrt(session_count(i) × session_count(j))
```

每个商品最多保存 200 个邻居。

### 8.2 召回阶段参数

- 使用最近 10 个去重商品作为种子；
- 种子位置权重：`exp(-0.1 × recency_index)`；
- 第 1 个种子权重 1，第 10 个约为 0.4066，不再使用下降过快的 `1/rank`；
- 将 10 个种子的邻居相似度加权累加；
- 最终取 Top200。

优化实验中，`seed10_exp01` 相比原 `seed5_reciprocal` 在 Top50 和 Top100 均有提升；Top100 HitRate 从 0.34656 提升到 0.35095，联合增益从 1.97 个百分点提升到 2.05 个百分点。

---

## 9. Transition 召回

### 9.1 有向多步转移

不同于对称 ItemCF，Transition 只统计从历史较早商品到较晚商品的方向：

```text
item_t -> item_(t+1), item_(t+2), item_(t+3)
```

只保留未来三步，且要求右侧事件时间严格晚于左侧事件；同商品自转移跳过。

### 9.2 转移权重

```text
weight = target_action_weight
       × [exp(-gap_minutes / 30) + 0.2 × exp(-gap_minutes / 1440)]
       / step_distance
```

其中行为权重为 1/3/6。公式同时表达：

- 30 分钟尺度的短期兴趣；
- 1 天尺度的弱长期兴趣；
- 越远的未来步贡献越小；
- 加购和购买转移权重更高。

### 9.3 查询参数

- 使用最近 5 个去重商品作为种子；
- 种子权重使用 `1/(recency+1)`；
- 每个商品索引最多 200 个转移邻居；
- 最终取 Top150。

多步长短期衰减版本在阶段实验中将单路 HitRate@50 从约 0.2590 提升到 0.2868；正式验证诊断达到 0.30792，新物品 HitRate 达到 0.35814。

---

## 10. Category 召回

### 10.1 Point-in-Time 类目热度

先取得每个事件发生时已经可见的商品类目，再在类目内部统计商品趋势分数：

```text
category_score = action_weight
               × [exp(-age_days / 7) + 0.2 × exp(-age_days / 30)]
```

每个类目最多保留 200 个候选商品。

### 10.2 查询阶段

- 使用最近 5 个去重历史商品；
- 取得这些商品在冻结时点的类目；
- 从对应类目的趋势商品池召回；
- 聚合分数：

```text
candidate_score += category_trend_score / [(seed_recency + 1) × category_rank]
```

- 最终取 Top150。

类目树实验还比较了父类目补充，但“精确类目趋势”整体优于过多父类目扩展，因此最终仍以当前类目的行为加权趋势热度为主体。Category 在新物品召回上有明显价值，统一 Top50 诊断的新物品 HitRate 为 0.30609，独占命中 681。

---

## 11. Item2Vec 召回

### 11.1 训练目标

使用 Session 商品序列训练 Skip-Gram Negative Sampling。中心商品与上下文商品形成正样本对，点击、加购、购买分别赋予 1/3/6 行为权重，样本对权重为中心行为与上下文行为权重的几何平均。

### 11.2 最终参数

| 参数 | 数值 |
|---|---:|
| Embedding 维度 | 128 |
| `min_count` | 5 |
| 负样本数 | 15 |
| Epoch | 10 |
| Batch Size | 4,096 |
| 初始学习率 | 0.01 |
| Subsample | `3e-5` |
| 采样分布指数 | 0.75 |
| 随机种子 | 2026 |

学习率调度：

- Epoch 1-5：0.01；
- Epoch 6-10：0.005；
- 若训练到 11-15：0.002。

### 11.3 自适应窗口与距离权重

每个 Session 的窗口半径：

```text
radius = min(10, max(5, round(sequence_length × 0.5)))
```

商品距离权重：

```text
distance <= 3: 1.0
distance > 3 : exp[-0.2 × (distance - 3)]
```

频率达到 `min_count=5` 的商品进入词表。高频商品按 Word2Vec 经典公式进行下采样；负样本按商品频次的 0.75 次方分布抽取。

### 11.4 Session 向量与 ANN

- 查询时使用最近 5 条历史事件中可被 Item2Vec 覆盖的商品；
- 位置权重：`1/sqrt(recency+1)`；
- 对商品向量加权平均并 L2 归一化；
- FAISS HNSW 内积检索，归一化后等价于余弦相似度；
- `M=32`，`efConstruction=120`，`efSearch=96`；
- 每个 Session 召回 Top250。

### 11.5 选择 Epoch 10 的依据

| Epoch | HitRate@250 | 高频目标 HitRate | 加权 HitRate |
|---:|---:|---:|---:|
| 5 | 0.57105 | 0.73727 | 0.74982 |
| 10 | **0.57121** | **0.73749** | 0.75018 |
| 15 | 0.57113 | 0.73738 | **0.75057** |

Epoch 15 只有极小的加权提升，但普通命中与高频目标命中不再提升，因此最终选择 Epoch 10。最终 Sampled AUC 和 Session GAUC 均约为 0.96249。

---

## 12. Two-Tower V2 召回

### 12.1 用户塔输入

对最多 30 个历史事件分别编码：

- 商品 ID Embedding：64 维；
- 行为类型 Embedding：16 维；
- 时间间隔桶 Embedding：16 维。

每个事件拼接后为 96 维：

```text
64 + 16 + 16 = 96
```

随后：

```text
96 -> Linear(96, 128) -> GRU(input=128, hidden=128)
```

同时计算历史商品 ID Embedding 的均值：

```text
mean_item_embedding: 64 -> Linear(64, 128)
```

GRU 最终隐藏状态与均值投影相加，再经过：

```text
LayerNorm(128) -> Linear(128, 64) -> L2 Normalize
```

最终用户向量为 64 维。

### 12.2 商品塔输入及维度变化

商品塔包含四组 64 维表示：

1. 商品 ID Embedding：64；
2. Item2Vec 128 维向量投影到 64；
3. 类目层级表示投影到 64；
4. 商品统计及可用状态投影到 64。

类目原始表示：

```text
leaf_category_embedding   = 32
parent_category_embedding = 32
root_category_embedding   = 32
depth_embedding           = 8
合计                       = 104
104 -> Linear(104, 64)
```

商品统计与可用状态：

```text
[连续统计特征 + availability_embedding(8)]
-> Linear(..., 64) -> GELU -> LayerNorm
```

四组 64 维拼接：

```text
64 × 4 = 256
```

最终商品 MLP：

```text
LayerNorm(256)
-> Linear(256, 128)
-> GELU
-> Dropout(0.1)
-> Linear(128, 64)
-> 与商品 ID Embedding 残差相加
-> L2 Normalize
```

### 12.3 96 个去重困难负样本

每个训练样本组合五类负样本池，并全局去重：

| 负样本来源 | 目标数量 |
|---|---:|
| Item2Vec ANN，跳过最相近的前 10 后选取 | 32 |
| ItemCF 与 Transition 交替 | 24 |
| Category | 16 |
| Hybrid Popular | 12 |
| 频次 0.75 次方随机采样 | 12 |
| 合计 | 96 |

若去重或屏蔽后不足 96，则继续按频次 0.75 次方随机补齐。屏蔽项包括：

- 当前真实目标；
- 当前 Session 在目标时刻及之后会出现的商品；
- 已经选过的负样本；
- 不在商品映射中的商品。

此外还使用 Batch 内目标作为负样本，并通过因果掩码过滤不合法比较。

### 12.4 训练参数

| 参数 | 数值 |
|---|---:|
| 最大 Epoch | 15 |
| Batch Size | 1,024 |
| FP16 | 开启 |
| 温度 | 0.07 |
| Early Stopping Patience | 4 |
| 困难负样本矩阵 | 414,481 × 96 |
| 验证召回 | Top300 |
| GPU | RTX 5060 8GB |

Item2Vec 参数在前两轮冻结，从第 3 轮开始以主学习率的 0.1 倍微调。

训练在第 10 轮触发早停，恢复综合选择得分最好的第 6 轮。总耗时约 3 小时 28 分钟。

验证结果：

- 七路联合 HitRate：0.80286；
- 加权联合 HitRate：0.94538；
- 最佳选择分数：0.80939。

---

## 13. 候选合并与 RRF 基线

### 13.1 去重合并

七路原始结果统一为：

```text
session, aid, source, source_rank, source_score
```

按 `(session, aid)` 聚合，保留：

- `source_count`：命中该商品的召回路线数量；
- `best_source_rank`：所有路线中的最佳名次；
- 每一路的 `source_rank_*`；
- 每一路的 `source_score_*`。

缺失路线的 rank 填 10,000，score 填 0，并额外生成 `source_present_*`，从而区分“未被该路召回”和“被召回但名次较深”。

### 13.2 RRF 公式

每一路对候选贡献：

```text
1 / (60 + source_rank)
```

七路相加：

```text
rrf_score(item) = sum_source 1 / (60 + rank_source(item))
```

RRF 只作为独立 Baseline 分数及一个可供精排学习的数值特征。最终候选构造明确设置：

```text
candidate_truncation = null
rrf_used_for_truncation = false
inject_missing_positives = false  # 最终验证/测试评估
```

### 13.3 候选规模

训练集完整候选：

| 项目 | 数值 |
|---|---:|
| 样本 | 414,481 |
| 去重候选行数 | 324,626,024 |
| 平均候选 | 783.21 |
| P50 | 782 |
| P90 | 910 |
| P99 | 991 |
| 最大 | 1,084 |

最终测试：

| 项目 | 数值 |
|---|---:|
| 样本 | 25,700 |
| 去重候选行数 | 18,109,917 |
| 平均候选 | 704.67 |
| Candidate HitRate | 0.79992 |

---

## 14. 精排特征

最终神经精排使用 38 个连续/二值特征和 2 个类别字段。

### 14.1 召回特征示例

- `rrf_score`；
- `source_count`；
- `best_source_rank`；
- `source_rank_recent`、`source_score_recent`；
- `source_rank_itemcf`、`source_score_itemcf`；
- `source_rank_item2vec`、`source_score_item2vec`；
- `source_rank_category`、`source_score_category`；
- `source_rank_transition`、`source_score_transition`；
- `source_rank_hybrid_popular`、`source_score_hybrid_popular`；
- `source_rank_two_tower`、`source_score_two_tower`；
- 七个 `source_present_*` 二值特征。

例子：某商品同时被 Item2Vec、双塔和 Category 召回，则 `source_count=3`；若双塔排名第 2、Item2Vec 第 8、Category 第 40，则 `best_source_rank=2`，对应各路线排名分别保留。

### 14.2 Session 特征示例

- `history_length`：历史事件数；
- `history_unique_items`：历史不同商品数；
- `session_span_minutes`：当前 Session 已持续分钟数；
- `last_type_id`：最近行为类型；
- `last_categoryid`：最近商品所属类目。

### 14.3 Session-候选交叉特征

- `history_candidate_count`：候选在历史中出现次数；
- `history_candidate_recency`：候选距离当前最近一次出现的位置；
- `candidate_is_last`：候选是否就是最后一个商品；
- `candidate_in_history`：候选是否在历史中出现；
- `same_category_as_last`：候选和最后商品是否同类目。

### 14.4 商品 Point-in-Time 特征

- `category_depth`；
- `in_tree`：类目是否存在于类目树；
- `available`：目标时刻以前最近一次可用状态；
- `candidate_age_days`：商品从首次出现到快照时刻的天数；
- `category_state_age_days`：类目状态已经持续的天数；
- `availability_state_age_days`：可用状态已经持续的天数。

连续特征中的缺失和无穷值先填 0，再用训练集均值与标准差执行 Z-Score。类别 ID 不作为连续数值输入。

### 14.5 类别 Embedding 与维度变化

原始连续输入为 38 维：

```text
continuous features = 38
last_categoryid embedding = 16
last_type_id embedding = 8
combined input = 38 + 16 + 8 = 62
```

未知或缺失类别映射为 padding/OOV 索引 0。

---

## 15. 多任务标签设计

最终监督目标采用累积序数标签：

| 真实下一行为 | Click 塔 | Cart 塔 | Order 塔 |
|---|---:|---:|---:|
| 点击 | 1 | 0 | 0 |
| 加购 | 1 | 1 | 0 |
| 购买 | 1 | 1 | 1 |
| 非目标候选 | 0 | 0 | 0 |

含义：购买是比加购、点击更强的正反馈。一个最终购买的商品同时应当被点击塔和加购塔视为正例。

三个塔在每个候选上都参与训练，不进行任务 Mask。损失为三任务 `BCEWithLogitsLoss`，并根据训练数据不平衡为每个任务设置正样本权重：

```text
pos_weight = min(50, sqrt(negative_count / positive_count))
```

最终训练得到的正样本权重约为：

```text
click: 27.99
cart : 50
order: 50
```

---

## 16. MMoE 结构与参数

输入为 62 维。

### 16.1 专家

- 专家数量：8；
- 每个专家：

```text
Linear(62, 64) -> LayerNorm(64) -> ReLU
```

### 16.2 门控与任务塔

- Click、Cart、Order 各有一个独立门控；
- 每个门控：`Linear(62, 8) -> Softmax`；
- 对 8 个专家输出进行任务相关加权；
- 每个任务塔：

```text
64 -> Linear(64, 32) -> ReLU -> Dropout(0.2) -> Linear(32, 1)
```

输出三个 Logit。

---

## 17. PLE 结构与参数

最终选择单层 PLE。

### 17.1 专家结构

- 共享专家：2 个；
- 每个任务专属专家：2 个；
- 三个任务共 6 个专属专家；
- 每个专家结构均为：

```text
Linear(62, 64) -> LayerNorm(64) -> ReLU
```

### 17.2 门控与任务塔

每个任务只混合：

```text
2 个共享专家 + 当前任务的 2 个专属专家
```

门控宽度为 4：

```text
Linear(62, 4) -> Softmax
```

每个任务塔：

```text
64 -> Linear(64, 32) -> ReLU -> Dropout(0.2) -> Linear(32, 1)
```

PLE 相比 MMoE 的核心区别是显式分离共享知识与任务专属知识，减少稀疏购买任务被大量点击信号淹没的问题。

---

## 18. 精排训练过程

| 参数 | 数值 |
|---|---:|
| 最大 Epoch | 15 |
| Early Stopping Patience | 4 |
| `min_delta` | 0.0002 |
| Batch Size | 4,096 |
| Parquet 流式批次 | 200,000 行 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | `1e-5` |
| 梯度裁剪 | 5.0 |
| FP16 | GPU 训练时开启 |
| 随机种子 | 2026 |

由于完整训练候选平均 783.21 个，训练时每个 Epoch 从每个样本的完整负例池中确定性变化采样约 160 个负例：

```text
negative_keep_probability = 160 / (783.21 - 1) ≈ 0.20455
```

负采样只使用 `(sample_id, aid, epoch_seed)` 哈希，和 RRF 分数无关。每个 Epoch 变化采样，使模型逐轮覆盖更多不同负例。

早停使用训练样本按时间顺序最后 10% 的内部切分，模型选择指标为加权 HitRate@20。真实最高分模型始终保存，`min_delta` 只控制耐心计数。

训练结果：

- MMoE：第 6 轮停止，恢复最佳第 4 轮；
- PLE：训练至第 9 轮，最佳第 9 轮；
- 两个模型 GPU 总耗时约 50 分钟。

冻结验证集：

| 模型 | HitRate@20 | 加权 HitRate@20 | NDCG@20 | MRR@20 | AUC | GAUC |
|---|---:|---:|---:|---:|---:|---:|
| MMoE | 0.60841 | 0.86462 | 0.42735 | 0.37427 | 0.88625 | 0.88845 |
| PLE | **0.61320** | **0.87360** | **0.43057** | **0.37695** | **0.89802** | **0.89954** |

因此最终选用 PLE 第 9 轮模型。

---

## 19. 三塔融合与最终排名

### 19.1 从 Logit 到概率

PLE 对每个候选输出：

```text
z_click, z_cart, z_order
```

先分别使用 Sigmoid 转换：

```text
p_click = sigmoid(z_click)
p_cart  = sigmoid(z_cart)
p_order = sigmoid(z_order)
```

### 19.2 冻结融合权重

验证集搜索后最终冻结：

```text
final_score = 0.05 × p_click
            + 0.30 × p_cart
            + 0.40 × p_order
```

权重不要求和为 1，因为这里只关心同一 Session 内的相对排序。点击权重较低、加购与购买权重较高，使前排更偏向高价值交互商品。

验证集选择融合权重时，前 50% 验证样本用于权重搜索，后 50% 用于独立 holdout 检查；权重冻结后再在完整验证集报告结果。测试集不参与权重选择。

### 19.3 统一列表生成

对每个 Session：

1. PLE 对所有完整去重候选打分；
2. 按 `final_score` 降序；
3. 分数相同时按 `aid` 升序，保证跨机器结果稳定；
4. 截取 Top5、Top10、Top20、Top50 评估；
5. 线上只输出一份统一商品列表，不输出三份行为列表。

RRF Baseline 使用同一批候选，直接按 `rrf_score` 排序。两者评估样本、候选集合和指标代码完全一致。

---

## 20. 指标定义

### 20.1 Candidate HitRate

真实下一商品是否存在于完整候选池：

```text
Candidate HitRate = hit_sessions / all_sessions
```

它衡量召回上限。

### 20.2 HitRate@K

每个 Session 只有一个目标商品，因此这里历史代码中的 `Recall@K` 与 HitRate@K 数值相同。最终文档统一称 HitRate：

```text
HitRate@K = mean(target_rank <= K)
```

### 20.3 NDCG@K 与 MRR@K

```text
NDCG@K = 1/log2(rank+1), rank<=K; otherwise 0
MRR@K  = 1/rank,         rank<=K; otherwise 0
```

二者不仅要求命中，还奖励更靠前的排名；MRR 对第一名附近的变化更敏感。

### 20.4 行为加权指标

真实目标行为的评估权重：

```text
click = 0.1, cart = 0.3, order = 0.6
```

注意：这是评估权重，不是三塔融合权重。融合权重为 0.05/0.30/0.40。

### 20.5 AUC、GAUC 与 UAUC

- Candidate AUC：在目标已被召回的 Session 中，目标分数战胜候选负例的比例；
- Session GAUC：先计算每个 Session 的 AUC，再求平均；
- End-to-End Session GAUC：未召回目标的 Session 按 0 计入；
- UAUC：按真实用户 `visitorid` 聚合 AUC；
- AUC 是全候选排序指标，不存在标准的 `AUC@5`。

---

## 21. 最终一次性测试

最终测试前冻结：

- Two-Tower V2：最佳第 6 轮；
- PLE：最佳第 9 轮；
- PLE 概率融合权重：0.05/0.30/0.40；
- 七路候选预算；
- RRF 常数 60；
- 评估 K：5、10、20、50。

测试完成标记确认：

```text
test_evaluated = true
model = ple_v3_epoch9
rrf_role = independent_baseline_only
```

### 21.1 候选结果

| 指标 | 数值 |
|---|---:|
| 测试 Session | 25,700 |
| 七路去重候选 | 18,109,917 |
| 平均候选 | 704.67 |
| Candidate HitRate | 0.79992 |
| 候选截断 | 无 |
| RRF 用于候选截断 | 否 |

### 21.2 PLE 与 RRF 头部排序

| 指标 | PLE | RRF | PLE 变化 |
|---|---:|---:|---:|
| HitRate@5 | **0.43455** | 0.41342 | **+2.11pp** |
| HitRate@10 | 0.50860 | **0.51198** | -0.34pp |
| HitRate@20 | 0.56938 | **0.59288** | -2.35pp |
| HitRate@50 | 0.61883 | **0.66669** | -4.79pp |
| NDCG@5 | **0.36213** | 0.29004 | **+7.21pp** |
| NDCG@10 | **0.38621** | 0.32208 | **+6.41pp** |
| NDCG@20 | **0.40163** | 0.34267 | **+5.90pp** |
| NDCG@50 | **0.41159** | 0.35740 | **+5.42pp** |
| MRR@5 | **0.33818** | 0.24912 | **+8.91pp** |
| MRR@10 | **0.34819** | 0.26244 | **+8.58pp** |

### 21.3 行为加权 HitRate

| 指标 | PLE | RRF | PLE 变化 |
|---|---:|---:|---:|
| Weighted HitRate@5 | **0.53056** | 0.47684 | **+5.37pp** |
| Weighted HitRate@10 | **0.59331** | 0.58080 | **+1.25pp** |
| Weighted HitRate@20 | 0.64978 | **0.65721** | -0.74pp |
| Weighted HitRate@50 | 0.69503 | **0.72411** | -2.91pp |

### 21.4 分行为命中

| K=5 | PLE | RRF |
|---|---:|---:|
| Click HitRate | **0.39680** | 0.38952 |
| Cart HitRate | **0.76100** | 0.59448 |
| Order HitRate | **0.85087** | 0.70174 |

PLE 在 Top5 对加购、购买目标的提升非常明显，解释了 Weighted HitRate、NDCG 和 MRR 的显著提升。

### 21.5 全候选区分指标

| 指标 | PLE | RRF |
|---|---:|---:|
| Conditional Candidate AUC | 0.87916 | **0.94425** |
| Conditional Session GAUC | 0.88169 | **0.94560** |
| End-to-End Session GAUC | 0.70528 | **0.75640** |
| Conditional UAUC | 0.88100 | **0.94634** |
| End-to-End UAUC | 0.72215 | **0.77578** |

---

## 22. 结果解释

PLE 并不是在所有指标上全面超过 RRF，而是形成清晰的头部价值取舍：

1. PLE 将高价值的加购和购买商品更积极地推入前 5 名；
2. 因此 PLE 的 HitRate@5、Weighted HitRate@5、NDCG@5、MRR@5 明显更好；
3. RRF 直接保留七路召回名次的强先验，在 Top20/50 覆盖和全候选 AUC 上更稳定；
4. PLE 使用 Pointwise BCE 训练，而最终关注的是 Top-K Listwise 指标，目标并不完全一致；
5. 固定三塔权重对不同 Session 的购买意图不能动态调整；
6. 精排训练负例是每 Epoch 采样，而测试需要面对平均 705 个完整候选，也会产生训练与推断难度差异。

因此最终结论不是“深度模型全面碾压规则”，而是：

> 当线上主要展示 5-10 个商品并重视加购、购买等高价值行为时，PLE 是更合适的主模型；当业务更关心 Top20-50 覆盖率和全候选排序稳定性时，RRF 仍是非常强的基线。

---

## 23. 项目最终结论

本项目完成了以下闭环：

- 面向 23.5 万商品的全量目录召回；
- 严格时间切分和 Point-in-Time 特征；
- 七路异构召回及单路优化；
- 128 维 Item2Vec 与 FAISS ANN；
- 融合序列、类目树、Item2Vec 和商品状态的双塔模型；
- 五类去重困难负样本；
- 不经 RRF 截断的完整候选精排；
- MMoE 与 PLE 多任务对比；
- 累积序数标签和统一 Top-K 输出；
- 独立 RRF 强基线；
- 冻结模型后一次性测试；
- HitRate、Weighted HitRate、NDCG、MRR、AUC、GAUC、UAUC 完整评估。

最终测试中，PLE 相对 RRF：

- HitRate@5 提升 2.11 个百分点；
- Weighted HitRate@5 提升 5.37 个百分点；
- NDCG@5 提升 7.21 个百分点；
- MRR@5 提升 8.91 个百分点。

实验阶段至此结束。后续若继续研究，应重新建立新的验证协议，不能根据本次测试集结果继续调参。可作为未来工作的方向包括：

- 用 PLE 学习 RRF 的残差，而非完全重排；
- Pairwise/Listwise Top-K 损失；
- 更严格的 OOF 候选训练；
- 根据 Session 上下文动态学习三塔融合权重；
- 对新物品和冷启动用户进行单独建模。

---

## 24. 关键产物与复现入口

- 数据预处理报告：`data/processed/retailrocket/manifest.json`
- 序列样本报告：`data/processed/retailrocket/sequence_samples_manifest.json`
- 商品属性报告：`data/processed/retailrocket/item_metadata_manifest.json`
- 七路诊断：`outputs/recall_route_summary/metrics.json`
- Item2Vec V2：`outputs/item2vec_v2_checkpoints/`
- 双塔 V2 验证：`outputs/windows_two_tower_v2_results_imported/`
- 精排 V3 验证：`outputs/windows_rankers_v3_results_imported/`
- 最终融合配置：`configs/ple_selected_fusion.json`
- 最终测试结果包：`windows_final_frozen_test_results.zip`（外部归档）
