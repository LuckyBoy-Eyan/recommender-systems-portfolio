# 多阶段会话推荐系统

本项目参考 Kaggle OTTO 推荐竞赛的多阶段架构，实现一个包含多路召回、特征工程、
Hard Negative Sampling、分目标排序和离线评估的端到端会话推荐系统。

项目重点关注：

- 如何构建无目标泄漏的离线实验；
- 如何评估每一路召回的覆盖能力；
- 如何针对点击、加购、购买设计差异化排序模型；
- 如何分析高指标背后的真实数据特性。

## 完整推荐链路

```text
Session 行为数据
  -> 按时间划分完整 Session + Leave-Last-Event-Out 标签
  -> 仅使用训练可见历史构建无泄漏召回图
  -> Recent + Popularity + ItemCF + Item2Vec 四路召回
  -> 候选集合并、RRF 排名融合基线与分召回源诊断
  -> 召回源、Session、物品、转化率及时间特征
  -> Hard Negative Sampling
  -> Shared-Bottom 多目标排序
  -> Weighted Recall@20
```

## 快速运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python scripts/run_pipeline.py
# 当前测试集已经用于历史实验，不要继续根据其结果选择参数
```

## 数据泄漏防护

完整 Session 首先按最后一个目标事件的时间划分为 70% 训练集、15% 验证集和 15%
测试集，随后执行 Leave-Last-Event-Out。常规运行只评估验证集；代码保留显式
`--evaluate-test` 开关，但当前测试集已经用于历史实验，后续调参不得反复查看。

每个目标按照自己的时间获得一个因果快照。RetailRocket 实验按天物化快照，所有全局
事件都满足 `event_ts < snapshot_ts <= target_ts`。Session 本地历史则可使用到目标发生
之前。快照中构建的内容包括：

- Popularity；
- ItemCF 相似度；
- Item2Vec 召回；
- 物品行为与转化率特征；
- 排序模型训练特征。

被留出的目标事件不会进入上述任何统计或特征。

当前 Top3000 目录是预处理阶段按全量事件频次确定的固定目录，因此属于
transductive catalog 设定；“严格 Point-in-Time”特指召回图、Item2Vec、商品统计和
排序特征的构建。若要求从目录选择开始也完全因果，应只用训练时间窗口确定 Top3000，
并重新统计可用 Session 和全部实验指标。

## RetailRocket 真实数据实验

RetailRocket 是公开真实电商行为数据集。本项目将事件类型映射为：

```text
view        -> clicks
addtocart   -> carts
transaction -> orders
```

可复现实验子集包含：

- Top3000 热门商品目录；
- 20,000 个无歧义目标时间的真实 Session；
- 2,802 个实际出现商品；
- 96,516 条用户行为；
- 87,293 次点击、6,547 次加购、2,676 次购买。

时间切分后得到 14,000/3,000/3,000 个训练、验证和测试 Session。当前配置使用：

```text
32维 / 窗口2 / 负样本5 / 8轮 / batch 1024 / 学习率0.01 / min_count 1
```

正式配置进一步使用点击/加购/购买 `1/3/6` Item2Vec 行为权重。Top3000/20,000
Session 新口径的锁定测试结果：

| 模型/阶段 | Candidate Recall | Weighted Candidate Recall | Weighted Recall@20 |
|---|---:|---:|---:|
| 合并候选上限 | 0.9310 | 0.9802 | — |
| RRF启发式融合 | — | — | 0.7770 |
| Shared-Bottom | — | — | **0.9668** |

Shared-Bottom相对RRF绝对提升`0.1898`、相对提升约`24.43%`。当前正式流水线
只训练Shared-Bottom，RRF作为无需训练的启发式基线。

下载 RetailRocket 数据后，可通过以下命令复现实验：

```bash
python scripts/prepare_retailrocket.py --events /path/to/events.csv
python scripts/run_pipeline.py --config configs/retailrocket.json \
  --output outputs/retailrocket_top3000_session20000_validation
# 当前测试集已完成一次锁定评估，后续不得根据测试结果继续调参
```

通用数据加载器同时支持 OTTO 风格的 Parquet、CSV 和 JSONL 数据。

详细实验协议、解释和局限见
[真实数据实验报告](docs/public-data-results.md)。

## 核心模块

- `src/data/load.py`：Parquet、CSV、JSONL 数据加载与字段标准化；
- `src/data/split.py`：按时间划分 Session 与构建预测标签；
- `src/recall/sources.py`：Recent、Popularity、ItemCF、Item2Vec 四路召回；
- `src/features/build.py`：候选、物品、Session、转化率及时间特征；
- `src/ranking/model.py`：Hard Negative与Shared-Bottom统一排序接口；
- `src/ranking/neural.py`：Shared-Bottom 与任务 Mask；
- `src/evaluation/metrics.py`：Candidate Recall、分召回源诊断及 Weighted Recall；
- `scripts/run_pipeline.py`：真实数据端到端训练与评估入口。

## 设计与评估要点

排序模型只能对已经被召回的候选进行优化，因此 Candidate Recall 决定了最终推荐
效果的上限。本项目会在训练排序模型前独立分析每一路召回，并使用真实召回候选中
未命中的商品作为 Hard Negative。

项目同时强调离线评估的时间边界：如果目标事件进入召回图、物品统计或特征，
离线指标会被严重高估，且无法反映真实线上效果。

当前正式链路不包含 Co-visitation。若后续重新加入，需要在当前数据口径上比较独占命中、
合并候选增量、排序收益和计算开销，不能沿用其他数据口径的消融结论。

启发式基线使用 Reciprocal Rank Fusion（RRF）统一各路召回的分数尺度：
候选在某一路排名为 \(r\) 时贡献 \(1/(60+r)\)，未被该路召回时贡献 0。它不再直接
相加量纲不同的 Popularity、ItemCF 和 Item2Vec 原始分数。

Item2Vec 使用 PyTorch 实现 Skip-Gram Negative Sampling。每个日级快照只用
`event_ts < snapshot_ts` 的 Session 序列重新训练，固定单线程、随机种子和负采样器。
当前参数为32维、窗口2、5个负样本、8轮训练、batch 1024和学习率0.01；参数只根据
验证集选择。正样本使用点击/加购/购买 `1/3/6` 权重。代码也实现了排除中心商品全部
真实上下文的负采样开关，但五种子消融没有稳定收益，因此正式配置关闭该开关。

当前正式测试中，Item2Vec 单路 Recall@20 为 `0.2197`。尚未在当前数据口径上完成
多随机种子 Item2Vec/SVD/无向量召回消融，因此不能归因其稳定增益。

Shared-Bottom使用共享底层和点击、加购、购买三个任务塔，并通过任务Mask避免把
未观测任务错误标为负例。测试Weighted Recall@20为`0.9668`。

更多内容：

- [文档导航](docs/README.md)
- [项目演进](docs/project-evolution.md)
- [代码导读与核心调用链](docs/code-walkthrough.md)
- [真实数据实验报告](docs/public-data-results.md)
- [Item2Vec 消融](docs/experiments/item2vec.md)
- [历史排序器实验](docs/experiments/ranker.md)
- [候选深度消融](docs/experiments/candidate-depth.md)
- [预热窗口消融](docs/experiments/warmup.md)
- [实验结果摘要](docs/results.md)
- [后续完善方向](docs/roadmap.md)

