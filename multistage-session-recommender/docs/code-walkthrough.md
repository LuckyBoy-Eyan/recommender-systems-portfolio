# 多阶段会话推荐系统：代码导读与核心调用链

这份文档按“第一次阅读项目代码”的顺序解释正式流水线。内容包括：

- 数据表在每个阶段的含义；
- 配置参数的用途；
- 主程序的核心调用链；
- 每个生产函数由谁调用、调用谁、参数和返回值是什么；
- 实验脚本和测试分别验证什么。

正式运行入口是 `scripts/run_pipeline.py`，当前正式配置是
`configs/retailrocket.json`。

## 1. 先认识四张核心数据表

### 原始标准事件表 `events`

| 列 | 含义 |
|---|---|
| `session` | 会话ID |
| `aid` | 商品ID |
| `ts` | 事件时间戳；RetailRocket中单位为毫秒 |
| `type` | `clicks`、`carts`或`orders` |

### 本地历史表 `history`

仍然是 `session/aid/ts/type` 四列，但每个Session的最后一个目标事件已经移除。
对每条历史必须满足 `history.ts < target_ts`。

### 标签表 `labels`

| 列 | 含义 |
|---|---|
| `session` | 会话ID |
| `target_aid` | 需要预测的最后一个商品 |
| `target_type` | 最后事件的行为类型 |
| `target_ts` | 最后事件时间 |

### 候选特征表 `features`

每个 `(session, aid)` 一行。除了标识列，还包括召回排名、召回分数、商品统计、
Session统计、Session-Item交叉特征和RRF分数，供排序模型训练和推理。

## 2. 当前正式配置

`configs/retailrocket.json` 的主要参数：

| 参数 | 含义 |
|---|---|
| `seed` | Item2Vec、Hard Negative和排序模型共同使用的随机种子 |
| `data_path` | 已完成Session化的标准事件文件 |
| `max_sessions` | 加载阶段最多保留多少Session；`null`表示全部 |
| `train_ratio` | 训练Session比例 |
| `valid_ratio` | 验证Session比例；剩余比例作为测试集 |
| `snapshot_interval` | Point-in-Time时间桶宽度；`86400000`毫秒为一天 |
| `candidates_per_source` | 每一路、每个Session最多返回多少候选 |
| `embedding` | Item2Vec构建与训练参数 |
| `rankers` | 唯一的Shared-Bottom排序器配置 |
| `primary_ranker` | 固定为`shared_bottom` |
| `topk` | 最终评估列表长度 |
| `max_negative_per_session` | 每个训练Session最多保留多少个召回负例 |

Item2Vec子配置：

| 参数 | 含义 |
|---|---|
| `method` | `item2vec`、`svd`或`none` |
| `max_neighbors` | 每个商品最多保存多少个向量近邻 |
| `dimensions` | 商品向量维度 |
| `window` | Skip-Gram左右上下文窗口 |
| `negative_samples` | 每个正样本配套的负商品数 |
| `epochs` | 每个快照训练轮数 |
| `batch_size` | Item2Vec训练批量大小 |
| `learning_rate` | Adam学习率 |
| `min_count` | 商品进入词表的最低出现次数 |
| `action_weights` | 点击、加购、购买正样本权重 |
| `exclude_positive_contexts` | 是否禁止把自身及全部已观察上下文抽成负例 |

当前正式配置只训练Shared-Bottom；RRF保留为无需训练的启发式基线。

Shared-Bottom子配置：

| 参数 | 含义 |
|---|---|
| `method` | 当前为`shared_bottom` |
| `hidden_dims` | 共享底层两层隐藏维度 |
| `epochs` | 排序器训练轮数 |
| `batch_size` | 排序器批量大小 |
| `learning_rate` | Adam学习率 |
| `weight_decay` | L2形式的权重衰减 |

## 3. 一眼看懂主调用链

```text
RetailRocket原始events.csv
  |
  v
scripts/prepare_retailrocket.py
  prepare_events
  |-- 行为映射
  |-- 30分钟切Session
  |-- Top3000商品过滤
  |-- 最短Session过滤
  `-- 删除歧义目标并递补到20000个Session
  |
  v
data/retailrocket/events.csv
  |
  v
scripts/run_pipeline.py::main
  |
  |-- load_events
  |-- drop_ambiguous_target_sessions
  |-- split_sessions
  |-- leave_last_event_out × 3
  |
  |-- build_point_in_time_dataset              [训练]
  |    |-- build_popularity
  |    |-- build_itemcf
  |    |-- build_embedding_neighbors
  |    |    `-- build_item2vec_neighbors
  |    |         |-- _build_skipgram_pairs
  |    |         `-- 可选_sample_negative_items
  |    |-- recall_candidates
  |    `-- build_candidate_features
  |
  |-- attach_labels
  |-- sample_hard_negatives
  |-- train_ranker_system
  |    `-- train_neural_ranker
  |         |-- NeuralFeaturePreprocessor.fit/transform
  |         |-- build_task_targets
  |         `-- _build_neural_model
  |              `-- SharedBottom
  |
  `-- evaluate_split                          [验证/可选测试]
       |-- build_point_in_time_dataset
       |-- heuristic_score
       |-- score_ranker_system × 3
       |-- evaluate_rankings × 3
       |-- candidate_recall
       `-- source_recall
```

## 4. 数据准备模块

### `scripts/prepare_retailrocket.py`

#### `prepare_events(events, catalog_size, max_sessions, min_session_length)`

- **谁调用**：同文件 `main`；单元测试也直接调用。
- **参数**
  - `events`：RetailRocket原始DataFrame，至少含
    `timestamp/visitorid/event/itemid`。
  - `catalog_size`：按行为频次保留的热门商品上限。
  - `max_sessions`：最终最多保留的Session数；删除歧义Session后从后续递补。
  - `min_session_length`：商品过滤后Session的最短行为数。
- **内部调用**：`drop_ambiguous_target_sessions`。
- **做什么**
  1. 行为类型映射；
  2. 按用户和时间排序；
  3. 相邻间隔大于30分钟时切新Session；
  4. Session化后再过滤热门商品；
  5. 过滤短Session和歧义目标；
  6. 按原始Session开始时间取最早的合格Session。
- **返回**
  - 标准事件表；
  - 数据规模和清洗参数组成的元数据字典。

#### `main()`

- **谁调用**：命令行直接运行该脚本。
- **做什么**：解析输入输出路径和清洗参数，调用 `prepare_events`，写出
  `events.csv` 与 `events.metadata.json`。

### `src/data/load.py`

#### `load_events(path, max_sessions=None)`

- **谁调用**：`scripts/run_pipeline.py::main`。
- **参数**
  - `path`：Parquet、CSV、JSONL或按行JSON文件路径。
  - `max_sessions`：可选Session数量限制。
- **做什么**：选择Pandas读取函数、检查四个必需字段、统一行为类型、过滤未知类型、
  可选截取最早Session，并按 `session/ts` 排序。
- **返回**：标准事件DataFrame。

### `src/data/split.py`

#### `drop_ambiguous_target_sessions(events, ts_column="ts")`

- **谁调用**：预处理脚本、主流水线入口和 `leave_last_event_out`。
- **参数**
  - `events`：含Session和时间列的事件表。
  - `ts_column`：时间列名；原始表用`timestamp`，标准表用`ts`。
- **做什么**：统计每个Session最大时间戳对应的事件数，删除计数大于1的整个Session。
- **返回**：无歧义Session事件表。

#### `leave_last_event_out(events)`

- **谁调用**：主流水线分别对训练、验证、测试事件调用。
- **内部调用**：`drop_ambiguous_target_sessions`。
- **做什么**：每个Session取唯一最后一行作为标签，其余作为历史，并再次过滤
  `history.ts >= target_ts` 的异常行。
- **返回**：`(history, labels)`。

#### `split_sessions(events, train_ratio, valid_ratio)`

- **谁调用**：主流水线入口。
- **参数**
  - `events`：完整Session事件表。
  - `train_ratio`：训练比例。
  - `valid_ratio`：验证比例。
- **做什么**：按每个Session最大时间排序，以严格时间条件构造训练、验证、测试边界；
  边界时间并列的Session整体进入较晚分区。
- **返回**：三个互不重叠的完整Session事件表。

## 5. Point-in-Time编排模块

### `src/features/point_in_time.py`

#### `build_point_in_time_dataset(...)`

- **谁调用**：主流水线构造训练样本，也由内部 `evaluate_split` 构造验证/测试样本。
- **参数**
  - `all_events`：当前阶段允许访问的全局事件池。
  - `sample_history`：待预测Session的本地历史。
  - `labels`：目标表。
  - `snapshot_interval`：时间桶宽度。
  - `candidates_per_source`：每路候选上限。
  - `seed`：向量模型随机种子。
  - `embedding_config`：Item2Vec/SVD/关闭向量召回的配置。
- **核心步骤**
  1. 审计本地历史是否严格早于目标；
  2. 把目标时间向下取整到快照起点；
  3. 每个快照只选择 `event.ts < snapshot_ts` 的全局事件；
  4. 构建Popularity、ItemCF和向量近邻；
  5. 执行四路召回；
  6. 构建排序特征；
  7. 保存快照审计信息。
- **内部调用**
  - `build_popularity`
  - `build_itemcf`
  - `build_embedding_neighbors`
  - `recall_candidates`
  - `build_candidate_features`
- **返回**
  - 候选特征宽表；
  - 带来源、排名、分数的召回长表；
  - 快照审计表。

## 6. 召回模块

### `src/recall/sources.py`

#### `build_popularity(events)`

- **谁调用**：Point-in-Time构建函数。
- **参数**：当前快照可见的历史事件。
- **做什么**：按商品事件次数降序排列。
- **返回**：热门商品ID列表。

#### `build_itemcf(events, max_neighbors=100)`

- **谁调用**：Point-in-Time构建函数。
- **参数**
  - `events`：当前快照历史。
  - `max_neighbors`：每个商品最多保存的邻居数。
- **做什么**：Session内商品去重，使用长Session降权统计共现，再用
  `sqrt(N_i*N_j)`归一化。
- **返回**：`{aid: [(neighbor, similarity), ...]}`。

#### `build_svd_neighbors(events, max_neighbors, dimensions, seed)`

- **谁调用**：`build_embedding_neighbors`在SVD消融配置下调用。
- **做什么**：建立行为加权Session-Item矩阵，执行TruncatedSVD，归一化后计算完整
  商品余弦近邻。
- **返回**：SVD商品近邻字典。

#### `_build_skipgram_pairs(events, item_to_index, window, action_weights)`

- **谁调用**：`build_item2vec_neighbors`。
- **参数**
  - `events`：当前快照历史；
  - `item_to_index`：原始商品ID到词表下标的映射；
  - `window`：左右上下文窗口；
  - `action_weights`：三种行为权重。
- **做什么**：把每个Session转成有序商品序列，枚举有序中心—上下文对，并计算
  `sqrt(center_weight*context_weight)`。
- **返回**：中心下标、上下文下标、训练对权重三个Tensor。

#### `_sample_negative_items(...)`

- **谁调用**：仅当 `exclude_positive_contexts=true` 时，由Item2Vec训练循环调用。
- **参数**
  - `center_batch`：当前批次中心商品下标；
  - `negative_distribution`：词表负采样概率；
  - `blocked_contexts`：每个中心禁止采样的商品矩阵；
  - `negative_samples`：每个中心所需负例数；
  - `generator`：固定种子的随机生成器。
- **做什么**：按分布抽样；命中自身或已屏蔽上下文时拒绝并重采样。
- **返回**：形状 `[B, negative_samples]` 的负商品下标。

#### `build_item2vec_neighbors(...)`

- **谁调用**：`build_embedding_neighbors`。
- **重要参数**：维度、窗口、负例数、轮数、批量、学习率、低频阈值、行为权重、
  是否排除真实上下文、随机种子。
- **内部调用**：`_build_skipgram_pairs`，可选 `_sample_negative_items`。
- **做什么**
  1. 构建词表；
  2. 生成带行为权重的Skip-Gram正样本；
  3. 从`count(aid)^0.75`分布采负例；
  4. 优化正负采样二元损失；
  5. L2归一化输入向量；
  6. 计算完整余弦Top-N近邻。
- **返回**：Item2Vec近邻字典。

#### `build_embedding_neighbors(events, method, ...)`

- **谁调用**：Point-in-Time构建函数。
- **做什么**：向量召回路由器：
  - `item2vec`调用`build_item2vec_neighbors`；
  - `svd`调用`build_svd_neighbors`；
  - `none`返回空字典。

#### `recall_candidates(...)`

- **谁调用**：Point-in-Time构建函数。
- **参数**
  - `history`：当前待预测Session历史；
  - `popularity`：热门列表；
  - `itemcf`：ItemCF近邻；
  - `per_source`：每路候选数；
  - `embedding`：Item2Vec或SVD近邻；
  - `embedding_source`：输出长表中的来源名。
- **做什么**
  - Recent取最近不同商品；
  - Popularity取热门列表头部；
  - ItemCF和向量召回使用最近5个不同商品作种子；
  - 多种子分数按新近性衰减后累加。
- **返回**：`session/aid/source/source_rank/source_score`召回长表。

## 7. 特征模块

### `src/features/build.py`

#### `build_candidate_features(history, recalled, reference_events=None)`

- **谁调用**：Point-in-Time构建函数。
- **参数**
  - `history`：Session本地历史；
  - `recalled`：多路召回长表；
  - `reference_events`：严格早于快照的全局历史。
- **做什么**
  1. 把召回长表透视成每个候选一行；
  2. 保留各路排名和分数；
  3. 统计召回源数量；
  4. 计算商品事件量、行为量、转化率；
  5. 计算Session和Session-Item交叉特征；
  6. 计算最佳来源排名和RRF。
- **返回**：缺失值填0的候选特征宽表。

## 8. 标签、负采样与统一排序接口

### `src/ranking/model.py`

#### `attach_labels(features, labels)`

- **谁调用**：主流水线训练阶段。
- **做什么**：把Session目标合并到候选表；`aid == target_aid`记为正例。
- **返回**：带`label/target_type/target_ts`的候选训练表。

#### `sample_hard_negatives(labeled, max_negatives, seed)`

- **谁调用**：主流水线训练阶段。
- **做什么**：每个Session保留全部召回正例，负例超过上限时随机截取。
- **注意**：这是“召回候选负例”，不是按上一版模型得分挖掘的最难负例。

#### `train_ranker_system(labeled, seed, ranker_config)`

- **谁调用**：主流水线。
- **做什么**：根据正式配置训练Shared-Bottom。
- **返回**：包含方法名和模型对象的排序系统字典。

#### `score_ranker_system(system, features, action)`

- **谁调用**：主流水线内部`evaluate_split`。
- **做什么**：调用Shared-Bottom对应任务塔输出候选分数。

#### `ranker_system_audit(system)`

- **谁调用**：主流水线写结果前。
- **做什么**：输出参数量、各任务观测行、正例数和类别权重。

#### `heuristic_score(features)`

- **谁调用**：主流水线内部`evaluate_split`。
- **做什么**：计算等权RRF基线：
  `sum(1 / (60 + source_rank))`。
- **返回**：按Session和RRF分数排序的`session/aid/score`表。

## 9. 神经多任务排序模块

### `src/ranking/neural.py`

#### `_needs_log1p(column)`

- **谁调用**：`NeuralFeaturePreprocessor.fit`。
- **做什么**：判断次数、长度、时间差等非负长尾列是否需要`log1p`。

#### `NeuralFeaturePreprocessor.fit(frame, columns)`

- **谁调用**：`train_neural_ranker`。
- **做什么**：只在训练候选上确定log列并拟合StandardScaler。
- **返回**：保存列顺序、log列和Scaler的预处理器。

#### `NeuralFeaturePreprocessor.transform(frame)`

- **谁调用**：神经训练和神经推理。
- **做什么**：按训练列顺序补齐特征，执行相同`log1p`与标准化。
- **返回**：`float32` NumPy特征矩阵。

#### `SharedBottom`

- 共享`input -> 64 -> 32`底层；
- 点击、加购、购买分别使用独立`32 -> 1`任务塔；
- 当前正式配置使用该模型。

`forward(features)`接收`[B, D]`，返回点击、加购、购买三个Logit，形状为`[B, 3]`。

#### `build_task_targets(labeled)`

- **谁调用**：`train_neural_ranker`。
- **做什么**：生成三任务标签与Mask。每行只让`target_type`对应任务的Mask为1，
  其他两个任务是未观测，而不是负例。
- **返回**：形状均为`[N, 3]`的标签Tensor和Mask Tensor。

#### `_build_neural_model(method, input_dim, hidden_dims)`

- **谁调用**：`train_neural_ranker`。
- **做什么**：校验`method=shared_bottom`并实例化SharedBottom。

#### `train_neural_ranker(...)`

- **谁调用**：`train_ranker_system`。
- **参数**：训练候选、特征列、结构名、随机种子、隐藏维度、轮数、批量、
  学习率、权重衰减。
- **内部调用**
  - 预处理器`fit/transform`；
  - `build_task_targets`；
  - `_build_neural_model`。
- **损失**
  - 每个任务只在Mask为1的行上计算BCEWithLogits；
  - `pos_weight = negatives / positives`；
  - 当前批次存在的任务损失等权平均。
- **返回**：模型、预处理器、类别权重、任务样本审计和可选Gate审计组成的Bundle。

#### `score_neural_candidates(bundle, features, action)`

- **谁调用**：`score_ranker_system`。
- **做什么**：复用训练预处理器，取得指定任务的Sigmoid概率，在每个Session内降序。
- **返回**：`session/aid/score`。

## 10. 评估模块

### `src/evaluation/metrics.py`

#### `evaluate_rankings(scored, labels, k)`

- **谁调用**：主流水线评估RRF基线和三个任务输出。
- **做什么**：每个Session取前K个商品，计算三类Recall@K，并按0.1/0.3/0.6加权。
- **注意**：调用方必须先把`scored`正确排序。

#### `source_recall(recalled, labels, k)`

- **谁调用**：主流水线。
- **做什么**：每个召回源独立取前K，统计它对全部标签Session的未加权命中率。

#### `candidate_recall(recalled, labels)`

- **谁调用**：主流水线。
- **做什么**：合并全部召回来源并去重，不截断Top-K、不区分任务，计算目标覆盖率。

#### `candidate_recall_by_action(recalled, labels)`

- **谁调用**：主流水线。
- **做什么**：分别计算点击、加购、购买候选覆盖，并按0.1/0.3/0.6得到
  Weighted Candidate Recall；它才是Weighted Recall@20可直接比较的候选上限。

三者不要混淆：

```text
Candidate Recall        = 召回阶段完整候选覆盖
Weighted Candidate Recall = 与最终指标同权重的候选覆盖上限
Source Recall@20        = 单一召回源前20覆盖
Weighted Recall@20      = 完整召回+指定任务排序后的最终前20指标
```

## 11. 主程序逐段阅读

`scripts/run_pipeline.py::main` 可以分成七段：

1. **读取配置与事件**
   - `load_events`
   - 防御性删除歧义Session
2. **构造数据集**
   - `split_sessions`
   - 三次`leave_last_event_out`
3. **构造训练候选**
   - `build_point_in_time_dataset`
4. **构造排序训练样本**
   - `attach_labels`
   - `sample_hard_negatives`
5. **训练排序器**
   - 调用`train_ranker_system`训练唯一的Shared-Bottom
6. **验证/可选测试**
   - 内部`evaluate_split`
   - RRF、三任务排序、三类指标
7. **保存产物**
   - `manifest.json`：配置、数据SHA-256、依赖版本、状态和耗时
   - `metrics.json`：验证/测试、分行为候选上限、分模型指标和阶段耗时
   - `ranker_systems.joblib`：Shared-Bottom模型与特征预处理器
   - Point-in-Time审计
   - 召回候选明细
   - 最多1000行训练样本

验证阶段的`reference_pool`只包含训练和验证阶段Session，不能访问测试Session。
测试阶段只有显式传入`--evaluate-test`才运行。

## 12. 实验入口

当前只保留两个可执行入口：

- `scripts/prepare_retailrocket.py`：清洗原始事件并生成标准化数据与元数据；
- `scripts/run_pipeline.py`：校验数据口径，训练当前配置并写出验证或最终评估产物。

测试集已经完成锁定评估，后续参数比较不得根据当前测试结果做选择。

## 13. 测试在验证什么

### `tests/test_pipeline.py`

- 最后事件是否正确留出；
- 最大时间戳并列时是否删除整个Session；
- 是否先Session化再过滤商品；
- 三个时间分区是否互斥且有严格时间边界；
- Point-in-Time是否排除未来事件；
- 目标事件是否进入召回图；
- Hard Negative是否保留正例并限制负例数；
- RRF是否不受原始分数量纲影响；
- Item2Vec在相同种子下是否确定；
- 行为权重是否正确进入Skip-Gram训练对；
- 开启排除策略时负例是否避开自身和已观察上下文。

### `tests/test_neural_rankers.py`

- 未观测任务是否通过Mask排除，而不是被标成负例；
- Shared-Bottom是否输出`[B, 3]`；
- Shared-Bottom训练和推理在相同种子下是否可复现。

## 14. 推荐阅读顺序

第一次学习项目时，建议按以下顺序：

1. `configs/retailrocket.json`
2. `scripts/run_pipeline.py`
3. `src/data/split.py`
4. `src/features/point_in_time.py`
5. `src/recall/sources.py::recall_candidates`
6. `src/recall/sources.py::build_item2vec_neighbors`
7. `src/features/build.py`
8. `src/ranking/model.py::train_ranker_system`
9. `src/ranking/neural.py::SharedBottom`
10. `src/ranking/neural.py::train_neural_ranker`
11. `src/evaluation/metrics.py`
12. 两个测试文件

先理解主链路，再进入Item2Vec和多任务模型细节，比从第一行顺序阅读所有文件更容易。
