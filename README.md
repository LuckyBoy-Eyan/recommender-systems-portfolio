# 推荐系统算法作品集

本仓库包含两个定位互补、可端到端复现的推荐系统项目：一个覆盖工业界常见的“召回-精排”链路，另一个探索 Semantic ID 自回归生成式推荐。

| 项目 | 技术重点 | 最终公开结果 |
|---|---|---|
| [多阶段会话推荐](multistage-session-recommender/) | 七路全量召回、Item2Vec、双塔困难负样本、PLE 多任务精排 | PLE 相对 RRF：HitRate@5 `+2.11pp`，NDCG@5 `+7.21pp` |
| [MiniTIGER](mini-tiger-generative-recommender/) | RQ-KMeans Semantic ID、自回归解码、SASRec 精排 | Recall@20 `0.2137`，NDCG@20 `0.1048` |

## 多阶段会话推荐亮点

- 在 RetailRocket 的 275 万条行为、140 万用户和 23.5 万事件商品上构建严格时间切分实验；
- Recent、Hybrid Popular、ItemCF、Item2Vec、Category、Transition、Two-Tower 七路召回；
- 128 维 Item2Vec、FAISS ANN、GRU 双塔与每样本 96 个去重困难负例；
- 38 个连续/二值特征加类目与行为 Embedding，比较 MMoE 和 PLE；
- 点击、加购、购买采用累积序数标签，三个任务分数融合为一份统一 Top-K 列表；
- RRF 仅作为独立强基线，不参与 PLE 候选截断；
- 模型和融合权重冻结后仅进行一次最终测试。

最终测试包含 25,700 个 Session，七路合并后平均 704.67 个候选，Candidate HitRate 为 `0.79992`。

| 指标 | PLE | RRF | 变化 |
|---|---:|---:|---:|
| HitRate@5 | **0.43455** | 0.41342 | **+2.11pp** |
| Weighted HitRate@5 | **0.53056** | 0.47684 | **+5.37pp** |
| NDCG@5 | **0.36213** | 0.29004 | **+7.21pp** |
| MRR@5 | **0.33818** | 0.24912 | **+8.91pp** |

RRF 在 Top20/50 覆盖率和全候选 AUC/GAUC/UAUC 上更强；项目保留这一真实结论，而不是将深度模型描述为全面领先。

## 仓库结构

```text
.
├── multistage-session-recommender/     # 七路召回 + PLE 多任务精排
├── mini-tiger-generative-recommender/  # Semantic ID 生成式推荐
├── README.md
└── .gitignore
```

原始数据、Parquet 候选、模型权重、训练输出和 ZIP 不提交至 Git。仓库保留源码、配置、测试、实验报告及轻量指标。

## 阅读顺序

1. [多阶段会话推荐 README](multistage-session-recommender/README.md)
2. [多阶段推荐完整实验报告](multistage-session-recommender/docs/complete-experiment-report.md)
3. [多阶段推荐项目演进](multistage-session-recommender/docs/project-evolution.md)
4. [MiniTIGER README](mini-tiger-generative-recommender/README.md)

## 本地检查

```bash
cd multistage-session-recommender
python -m pytest -q

cd ../mini-tiger-generative-recommender
python -m pytest -q
```
