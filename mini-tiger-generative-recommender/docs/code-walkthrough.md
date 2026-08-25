# MiniTIGER 全代码解读

这份文档按程序真实执行顺序解释代码。阅读时建议同时打开
`scripts/run_demo.py`，它是整个项目的总入口。

项目现已支持两套协议：旧合成/MovieLens 实验用于快速回归；KuaiRec 主实验使用
`prepare_kuairec.py`、逐用户时间切分、验证集早停以及 Trie Beam Search。完整
KuaiRec 设计与数据统计见 [KuaiRec 实验](kuairec-experiment.md)。

## 1. 这个项目到底“生成”什么

模型不生成自然语言，而是生成下一个物品的离散 Semantic ID。

假设某件物品的完整编码为 `[3, 1, 0]`，模型计算：

```text
P(item | history)
  = P(c0=3 | history)
  × P(c1=1 | history, c0=3)
  × P(tail=0 | history, c0=3, c1=1)
```

最后一级依赖前面已经生成的 token，所以这是自回归生成。完整编码再通过商品
目录映射回具体 item。

## 2. 顶层调用链

```text
scripts/run_demo.py::main
│
├── 读取 configs/*.json
│
├── 数据（二选一）
│   ├── src.data.synthetic.make_synthetic_data
│   └── src.data.load.load_interactions
│
├── Semantic ID
│   ├── src.indexing.rq_kmeans.build_rq_kmeans_codes（主路径）
│   ├── src.indexing.rq_kmeans.save_rq_kmeans_artifact
│   ├── src.indexing.semantic_ids.build_hierarchical_codes（旧消融）
│   ├── src.indexing.semantic_ids.append_collision_token
│   └── src.indexing.semantic_ids.collision_rate
│
├── src.data.dataset.NextItemDataset
├── src.data.split.temporal_leave_two_out（KuaiRec）
│
├── src.models.generative.SemanticIDTransformer
│   └── forward
│
├── src.training.train.train_model
│   └── model.forward(history, target_codes)
│
├── src.training.evaluate.evaluate_model
│   ├── exact：encode_history 一次 + 分块 decode 候选
│   ├── beam：合法前缀 Trie + constrained_beam_search
│   └── src.evaluation.metrics.ranking_metrics
│
├── src.training.evaluate.evaluate_popularity
│   └── src.evaluation.metrics.ranking_metrics
│
├── build_random_codes 后重复数据集、模型、训练和评估
│
└── 写入 outputs/<experiment>/metrics.json
```

## 3. `scripts/run_demo.py`：实验总入口

### `main()`

- Python 参数：无。
- 命令行参数：
  - `--config`：实验配置文件，默认 `configs/demo.json`。
  - `--output`：结果目录，默认 `outputs/demo`。
- 调用者：文件末尾的脚本入口。
- 调用的项目函数：
  - `load_interactions` 或 `make_synthetic_data`
  - `build_rq_kmeans_codes` 或旧消融 `build_hierarchical_codes`
  - `append_collision_token`
  - `NextItemDataset`
  - `SemanticIDTransformer`
  - `train_model`
  - `evaluate_model`
  - `build_random_codes`
  - `evaluate_popularity`
  - `collision_rate`
- 作用：固定随机种子，编排两组模型实验，汇总并保存指标。
- 返回值：无；结果写入 JSON 并打印到终端。

Semantic ID 和 Random ID 使用相同的网络容量与训练超参数。这样两者的主要差异
是输出编码是否具有物品语义，而不是模型大小不同。

## 4. `src/data/synthetic.py`：合成数据

### `make_synthetic_data(...)`

参数：

| 参数 | 含义 |
|---|---|
| `num_users` | 用户数量 |
| `num_items` | 物品数量 |
| `num_topics` | 隐藏兴趣主题数量 |
| `feature_dim` | 物品特征维度 |
| `min_seq_len` | 用户序列最短长度 |
| `max_seq_len` | 用户序列最长长度 |
| `seed` | 随机种子 |

返回：

- `item_features`：形状 `[num_items, feature_dim]`。
- `sequences`：每个用户一条 item index 序列。

调用者：`run_demo.main`，仅在配置没有真实数据路径时调用。

工作过程：

1. 随机生成多个 topic 中心。
2. 给每个 item 分配一个 topic。
3. 用“topic 中心 + 小噪声”生成 item 特征。
4. 给每个用户分配两个偏好 topic。
5. 用户以 `0.75/0.25` 的概率从两个 topic 中采样物品。

因此，物品特征中的相似性与用户行为中的兴趣结构一致，Semantic ID 才有机会
优于 Random ID。它是链路冒烟数据，不代表真实业务数据。

## 5. `src/data/load.py`：真实数据读取

### `load_interactions(...)`

参数：

| 参数 | 含义 |
|---|---|
| `interactions_path` | 包含 `user_id,item_id,timestamp` 的 CSV |
| `item_features_path` | 与排序后 item ID 对齐的 NumPy 特征文件 |
| `min_sequence_length` | 用户行为序列最短保留长度，默认 3 |

返回：

- `features`：float32 数组，形状 `[num_items, feature_dim]`。
- `sequences`：按用户分组、按时间排序的连续 item index 序列。

调用者：`run_demo.main`。

关键点：

- 原始 item ID 可能不是连续整数，所以函数先排序并映射到 `0...N-1`。
- `item_features.npy` 的第 i 行必须对应排序后第 i 个 item ID。
- CSV 缺列或特征行数不匹配时会主动抛出 `ValueError`。

## 6. `scripts/prepare_movielens.py`：MovieLens 预处理

### `main()`

命令行参数：

| 参数 | 含义 |
|---|---|
| `--source` | 包含 MovieLens 原始 CSV 的目录 |
| `--output` | 输出目录 |
| `--catalog-size` | 按正反馈次数保留的电影数 |
| `--min-rating` | 正反馈最低评分 |
| `--min-sequence-length` | 用户最少正反馈数 |

输出：

- `interactions.csv`：统一字段名后的正反馈行为。
- `item_features.npy`：电影 genre multi-hot 特征，经行归一化。

调用者：用户从命令行单独运行。它不参与每次模型训练，只负责提前准备数据。

## 7. `src/indexing/rq_kmeans.py`：工业物品离散编码

### `build_rq_kmeans_codes(features, codebook_sizes, seed, ...)`

- `features`：`[num_items, feature_dim]` 连续物品特征。
- `codebook_sizes`：各级全局残差码本大小。
- `seed`：PCA 和 K-Means 的可复现种子。
- `backend`：`auto` 优先 Faiss，或强制 `faiss` / `sklearn`。
- `pca_dim`、`whiten`、`l2_normalize`：聚类前特征预处理。
- `niter`、`nredo`：K-Means 迭代和重启次数。
- `use_gpu`：是否让 Faiss 使用 CUDA。
- `max_balance_ratio`：单个 token 桶相对平均桶的容量上限。
- `resolve_collisions`：是否对末级碰撞做最小代价唯一匹配。
- 返回：`RQKMeansResult`，含 codes、centroid、诊断和索引指纹。
- 调用者：`run_demo.main`。

执行逻辑：

```text
PCA/白化/归一化(item_features) -> residual_0
KMeans(residual_0) -> 容量约束 token_0 -> residual_1
KMeans(residual_1) -> 容量约束 token_1 -> residual_2
...
共享前缀碰撞 -> 末级 Hungarian 最小代价匹配
```

### `save_rq_kmeans_artifact(result, output_dir)`

- `result`：上述构建结果。
- `output_dir`：索引发布目录。
- 作用：保存可重放的 PCA 数组、每层 centroid、codes、Schema 和 SHA-256 指纹。
- 返回：无。

## 8. `src/indexing/semantic_ids.py`：旧索引消融与通用编码工具

### `build_hierarchical_codes(features, codebook_sizes, seed)`

- `features`：`[num_items, feature_dim]` 连续物品特征。
- `codebook_sizes`：每级聚类数，例如 `[8, 8]`。
- `seed`：KMeans 随机种子。
- 返回：`[num_items, num_levels]` 整数编码矩阵。
- 调用者：`run_demo.main`。

当 `semantic_id_method` 设置为 `residual` 时执行旧全局残差逻辑：

```text
residual_0 = item_features
第 0 级：KMeans(residual_0) -> c0
residual_1 = residual_0 - 第 0 级簇中心
第 1 级：KMeans(residual_1) -> c1
...
code = [c0, c1, ...]
```

第一级表达粗粒度结构，后续层表达此前没有解释掉的细节。

### `collision_rate(codes)`

- `codes`：编码矩阵。
- 返回：`1 - 唯一编码数 / 物品数`。
- 调用者：`run_demo.main` 和单元测试。
- 作用：判断完整编码能否唯一定位商品。

### `append_collision_token(codes)`

- `codes`：可能重复的 Semantic ID 前缀。
- 返回：
  - 追加 tail 后的编码矩阵；
  - tail 层词表大小。
- 调用者：`run_demo.main`、`build_random_codes` 和单元测试。

例如：

```text
[2, 5] -> [2, 5, 0]
[2, 5] -> [2, 5, 1]
[4, 3] -> [4, 3, 0]
```

前缀仍保留共享语义，tail 负责区分具体物品。

### `build_random_codes(num_items, codebook_sizes, seed)`

- `num_items`：物品总数。
- `codebook_sizes`：各层随机 token 的范围。
- `seed`：随机种子。
- 返回：
  - 已追加 tail 的随机编码；
  - 包含 tail 大小的完整词表大小列表。
- 调用者：`run_demo.main`。
- 内部调用：`append_collision_token`。

它是消融对照：编码层数和模型容量相近，但编码前缀不携带内容语义。

## 8. `src/data/dataset.py`：监督样本

### `NextItemDataset.__init__(...)`

参数：

| 参数 | 含义 |
|---|---|
| `sequences` | 用户 item index 序列 |
| `item_codes` | `item index -> 完整编码` 查找表 |
| `max_history` | 最多保留最近多少个历史物品 |
| `last_only` | 是否每个用户只预测最后一个物品 |

调用者：`run_demo.main`。

序列 `[A, B, C, D]` 在普通模式下产生：

```text
[A, B]    -> C
[A, B, C] -> D
```

在 `last_only=True` 时只产生第二条，常用于 leave-one-out 评估。

### `__len__()`

- 参数：只有 Python 自动传入的 `self`。
- 返回：样本数量。
- 调用者：PyTorch `DataLoader`、入口中的实验统计和评估函数。

### `__getitem__(index)`

- `index`：样本下标。
- 返回：
  - `history_codes`：`[max_history, num_levels]`；
  - `target_codes`：`[num_levels]`；
- `target_item`：原始 item index 标量。
- `history_items`：定长原始历史 item index，左侧以 -1 padding；评估时用于
  排除用户已经看过的视频。
- 调用者：PyTorch `DataLoader`，通常不手动调用。

历史 token 全部加 1，把 0 留给 padding。目标 token 不加 1，因为输出分类头的
标签范围本来就是 `0...vocab_size-1`。

## 9. `src/models/generative.py`：模型

### `SemanticIDTransformer.__init__(...)`

参数：

| 参数 | 含义 |
|---|---|
| `codebook_sizes` | 各级 token 词表大小，包含 tail |
| `max_history` | 历史序列固定长度 |
| `hidden_dim` | embedding、Transformer 和 GRU 状态维度 |
| `num_heads` | Transformer 注意力头数 |
| `num_layers` | Transformer Encoder 层数 |

调用者：`run_demo.main`。

主要子模块：

- `code_embeddings`：历史物品每一级编码各有一张 embedding 表。
- `position`：历史位置 embedding。
- `encoder`：Transformer Encoder，理解行为序列。
- `heads`：每一级 token 一个分类头。
- `target_embeddings`：把上一层 token 变成下一解码步输入。
- `decoder_cell`：GRUCell，负责层级间自回归状态更新。
- `start_token`：第一级生成前的可学习起始向量。

### `forward(history_codes, target_codes=None)`

参数：

- `history_codes`：`[batch_size, max_history, num_levels]`。
- `target_codes`：可选，`[batch_size, num_levels]`。

返回：

- `logits` 列表；第 l 个元素形状是
  `[batch_size, codebook_sizes[l]]`。

调用者：

- `train_model`：传真实目标编码，进行 Teacher Forcing。
- `evaluate_model`：传某个目录候选编码，计算这条编码的精确条件概率。
- 单元测试：检查后一级输出是否依赖前一级 token。

内部过程：

1. 对历史中每件物品的各级 code embedding 求和。
2. 加上历史位置 embedding。
3. mask 左侧 padding，送入 Transformer。
4. 取最后一个历史位置作为用户当前兴趣状态。
5. 从 `start_token` 开始，多次调用 GRUCell。
6. 每一级分类头预测当前 token。
7. 当前 token 的 embedding 成为下一层解码输入。

传 `target_codes` 时使用真实前缀；不传时使用模型当前层的 argmax。这一差异就是
Teacher Forcing 与自由贪心生成的区别。

## 10. `src/training/train.py`：训练

### `train_model(model, dataset, epochs, batch_size, learning_rate)`

参数：

| 参数 | 含义 |
|---|---|
| `model` | `SemanticIDTransformer` |
| `dataset` | `NextItemDataset` |
| `epochs` | 完整遍历数据的次数 |
| `batch_size` | 单次更新的样本数 |
| `learning_rate` | AdamW 学习率 |

- 返回：训练后的同一个模型对象。
- 调用者：`run_demo.main`。
- 内部调用：`DataLoader`、`model.forward`、交叉熵、反向传播和 AdamW。

损失函数：

```text
loss = CE(c0_logits, c0_target)
     + CE(c1_logits, c1_target)
     + ...
     + CE(tail_logits, tail_target)
```

训练只需要 `target_codes`，不需要原始 `target_item`。

## 11. `src/training/evaluate.py`：离线评估

### `evaluate_model_exact(model, dataset, item_codes, ks, ...)`

- `model`：训练好的模型。
- `dataset`：测试数据集。
- `item_codes`：完整合法商品编码目录。
- `ks`：例如 `[5, 10, 20]`。
- 返回：Recall、HitRate、NDCG、MRR 字典。
- 调用者：`evaluate_model` 分派器。
- 内部调用：`model.encode_history`、`model.decode` 和 `ranking_metrics`。

函数先把每条用户历史编码一次，再把候选目录分块，仅展开轻量 GRU 解码状态。
候选 item 的分数是：

```text
score(item)
  = log P(c0 | history)
  + log P(c1 | history, c0)
  + ...
```

这里传入候选编码不是泄露答案，因为每个候选都被同样打分；它是在计算每条候选
路径自身的条件概率。最后比较所有路径的概率并排序。

该方法保证只推荐合法商品，适合作为精确基准，但复杂度仍随目录线性增长。

### `build_prefix_trie(item_codes)`

把完整商品编码整理为“当前前缀 -> 下一步合法 token”，并建立完整编码到 item
index 的映射。

### `constrained_beam_search(...)`

为单个用户逐级扩展概率最高的合法编码前缀。每一级只访问 Trie 允许的 token，
最终结果必然是目录中的真实视频。

### `evaluate_model_beam(...)`

批量编码用户历史，再逐用户运行约束 Beam Search，适用于 KuaiRec 万级目录。

### `evaluate_model(...)`

根据配置中的 `evaluation_mode` 分派到 `exact` 或 `beam`。

### `evaluate_popularity(dataset, train_sequences, ks)`

- `dataset`：提供真实测试目标。
- `train_sequences`：统计 item 出现频次。
- `ks`：指标截断位置。
- 返回：与模型评估相同结构的指标。
- 调用者：`run_demo.main`。
- 内部调用：`ranking_metrics`。

所有测试用户得到同一份热门榜，是最简单的非个性化 baseline。

## 12. `src/evaluation/metrics.py`：指标

### `ranking_metrics(rankings, targets, ks)`

- `rankings`：每条样本的 item 排序列表。
- `targets`：每条样本唯一的真实下一个 item。
- `ks`：截断位置。
- 返回：各个 k 上的 Recall、HitRate、NDCG、MRR。
- 调用者：`evaluate_model` 和 `evaluate_popularity`。

由于每条样本只有一个真实目标，本项目中的 Recall@K 和 HitRate@K 相等：

```text
Recall/HitRate = 是否在前 K 命中
NDCG           = 命中时 1 / log2(rank + 1)
MRR            = 命中时 1 / rank
```

## 13. 配置文件

JSON 不允许注释，因此参数统一在这里解释。

| 配置项 | 含义 |
|---|---|
| `seed` | Python、NumPy、PyTorch 和 KMeans 的可复现种子 |
| `interactions_path` | 真实交互 CSV；为 null 时使用合成数据 |
| `item_features_path` | 真实物品特征 NPY；为 null 时使用合成数据 |
| `eval_last_only` | 是否每个测试用户只评估最后一次行为 |
| `num_users` | 合成用户数 |
| `num_items` | 合成物品数 |
| `num_topics` | 合成数据隐藏主题数 |
| `feature_dim` | 合成物品特征维度 |
| `min_seq_len` | 合成用户序列最短长度 |
| `max_seq_len` | 合成用户序列最长长度 |
| `codebook_sizes` | 不含 tail 的各级编码词表大小 |
| `max_history` | 最长历史行为数 |
| `hidden_dim` | 模型隐藏维度 |
| `num_layers` | Transformer Encoder 层数 |
| `num_heads` | Transformer 注意力头数 |
| `batch_size` | 训练 batch 大小 |
| `epochs` | 训练轮数 |
| `learning_rate` | AdamW 学习率 |
| `topk` | 需要输出指标的 K 值 |
| `split_strategy` | KuaiRec 使用 `temporal_leave_two_out` |
| `max_sequence_length` | 每用户最多保留最近多少条行为 |
| `max_train_samples_per_user` | 每用户最多贡献多少个训练目标 |
| `device` | `auto`、`cpu`、`mps` 或 `cuda` |
| `evaluation_mode` | `exact` 或 `beam` |
| `catalog_chunk_size` | 精确评估的候选分块大小 |
| `beam_size` | Trie Beam Search 保留的前缀数量 |
| `exclude_seen_items` | 是否从推荐结果排除历史物品 |
| `validate_every_epoch` | 是否每轮计算验证指标 |
| `monitor_metric` | 选择最佳模型的验证指标 |
| `early_stopping_patience` | 连续多少轮无提升后停止 |

## 14. 测试文件

`conftest.py` 在 pytest 启动时自动加载，限制数学库线程数，减少小环境并行冲突。

`tests/test_semantic_ids.py` 包含四项核心测试：

1. `test_semantic_id_shape_and_collision_range`：层次编码形状和碰撞率合法。
2. `test_collision_token_makes_catalog_ids_unique`：tail 能消除碰撞。
3. `test_decoder_is_conditioned_on_previous_target_tokens`：验证模型确实自回归。
4. `test_last_only_evaluation_has_one_sample_per_sequence`：leave-one-out 切分正确。

## 15. 建议的实际阅读顺序

1. `configs/demo.json`：先知道 Demo 有多大。
2. `scripts/run_demo.py`：只读主干调用。
3. `src/data/synthetic.py`：理解输入从哪里来。
4. `src/indexing/semantic_ids.py`：理解物品如何变成 token。
5. `src/data/dataset.py`：理解一条训练样本。
6. `src/models/generative.py`：跟踪所有张量形状。
7. `src/training/train.py`：理解 Teacher Forcing 和 loss。
8. `src/training/evaluate.py`：理解缓存精确打分与 Trie Beam Search。
9. `src/evaluation/metrics.py`：理解最终数字。
10. `tests/test_semantic_ids.py`：反向确认核心设计约束。

读完整条链路后，可以用一句话概括项目：

> MiniTIGER 先通过残差 KMeans 把物品特征量化为具有共享语义前缀的离散 ID，
> 再用 Transformer 编码用户历史、用 GRU 自回归生成下一个物品的多级 ID，
> 最后在合法商品目录上按完整编码概率排序。
