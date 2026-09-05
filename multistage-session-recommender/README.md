# 多阶段会话推荐系统

基于 RetailRocket 的全量目录 Session 推荐项目：七路召回从约 23.5 万件商品中产生候选，PLE 多任务精排输出一份统一 Top-K 商品列表，并以相同候选上的 RRF 作为独立强基线。

## 最终链路

```text
RetailRocket 原始事件与商品属性
  -> 30 分钟切分 Session
  -> 严格时间切分 + 滑动前缀样本
  -> Point-in-Time 召回索引与商品状态
  -> Recent / Hybrid Popular / ItemCF / Item2Vec
     / Category / Transition / Two-Tower 七路召回
  -> 按 (session, item) 完整去重，不使用 RRF 截断
  -> 召回、Session、商品和交叉特征
  -> MMoE / PLE 累积标签多任务精排
  -> 0.05 click + 0.30 cart + 0.40 order
  -> 统一 Top5 / Top10 / Top20 / Top50
```

## 数据与协议

| 项目 | 数量 |
|---|---:|
| 去重事件 | 2,755,641 |
| 用户 | 1,407,580 |
| 事件商品 | 235,061 |
| 滑动前缀训练样本 | 414,481 |
| 验证样本 | 25,698 |
| 测试样本 | 25,700 |

- 目标时刻前至少保留 2 个历史事件，历史最多取 30；
- 过短或末时间戳并列的 Session 不参与监督，但保留作历史统计；
- 所有召回图、Item2Vec 和商品状态满足 `feature_ts < target_ts`；
- 测试集在双塔、PLE、候选配额和融合权重冻结后只评估一次；
- 最终测试不强行注入漏召回目标。

## 七路召回

| 路线 | 预算 | 核心方法 |
|---|---:|---|
| Recent | 30 | 最近不同商品，`1/rank` |
| Hybrid Popular | 100 | 行为分层、时间趋势、新品和 Session 阶段配额 |
| ItemCF | 200 | Session 共现余弦，最近 10 个种子，慢指数衰减 |
| Item2Vec | 250 | 128 维 SGNS，FAISS HNSW ANN |
| Category | 150 | 类目内行为加权与 7/30 天时间衰减 |
| Transition | 150 | 未来三步有向转移，30 分钟/1 天长短期衰减 |
| Two-Tower | 300 | GRU 用户塔 + Item2Vec/类目树/状态商品塔 |

理论预算 1,180，去重后测试集平均 704.67 个候选。RRF 为：

```text
rrf_score(item) = sum_source 1 / (60 + rank_source(item))
```

RRF 是独立基线，不负责筛选 PLE 候选。

## Item2Vec 与双塔

Item2Vec 最终参数：128 维、`min_count=5`、15 个负样本、Epoch 10、Batch 4096、`subsample=3e-5`、自适应窗口 5-10、频次 0.75 次方负采样。

双塔 V2：

- 用户塔：商品/行为/时间 Embedding -> 128 维 GRU -> 64 维归一化向量；
- 商品塔：商品 ID、Item2Vec、叶子/父/根类目、深度、可用状态及统计特征 -> `256 -> 128 -> 64`；
- 每个样本 96 个去重困难负例，Batch 1024，FP16；
- RTX 5060 训练约 3 小时 28 分钟，第 10 轮早停，恢复第 6 轮。

## PLE 多任务精排

连续/二值输入 38 维，另加入 `last_categoryid` 16 维 Embedding 与 `last_type_id` 8 维 Embedding，总输入约 62 维。

| 真实行为 | Click 塔 | Cart 塔 | Order 塔 |
|---|---:|---:|---:|
| 点击 | 1 | 0 | 0 |
| 加购 | 1 | 1 | 0 |
| 购买 | 1 | 1 | 1 |

PLE 使用 2 个共享专家、每任务 2 个专属专家；专家为 `Linear(62,64) -> LayerNorm -> ReLU`，任务塔为 `64 -> 32 -> ReLU -> Dropout(0.2) -> 1`。最终选择第 9 轮模型。

## 一次性最终测试

测试候选：18,109,917 行、25,700 个 Session、平均 704.67 个候选，Candidate HitRate `0.79992`；`candidate_truncation=null`、`rrf_used_for_truncation=false`。

| 指标 | PLE | RRF | PLE 变化 |
|---|---:|---:|---:|
| HitRate@5 | **0.43455** | 0.41342 | **+2.11pp** |
| HitRate@10 | 0.50860 | **0.51198** | -0.34pp |
| HitRate@20 | 0.56938 | **0.59288** | -2.35pp |
| HitRate@50 | 0.61883 | **0.66669** | -4.79pp |
| Weighted HitRate@5 | **0.53056** | 0.47684 | **+5.37pp** |
| NDCG@5 | **0.36213** | 0.29004 | **+7.21pp** |
| NDCG@10 | **0.38621** | 0.32208 | **+6.41pp** |
| MRR@5 | **0.33818** | 0.24912 | **+8.91pp** |

PLE 更适合强调前 5-10 位和高价值行为的场景；RRF 在 Top20/50 覆盖及 AUC、GAUC、UAUC 上更稳定。

## 快速检查

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

原始数据、模型权重和完整候选不进入 Git。详细复现顺序、参数、公式、特征及结果见[完整实验报告](docs/complete-experiment-report.md)。

## 文档

- [完整实验报告](docs/complete-experiment-report.md)
- [结果摘要](docs/results.md)
- [项目演进](docs/project-evolution.md)
- [代码导读](docs/code-walkthrough.md)
- [历史实验归档](docs/experiments/README.md)
