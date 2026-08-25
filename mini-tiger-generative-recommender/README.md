# MiniTIGER：生成式多阶段推荐系统

这是一个面向 KuaiRec Big Matrix 的生成式序列推荐项目。系统不直接预测 Item ID，
而是根据用户历史自回归生成物品的多级 Semantic ID，再由 SASRec 对生成候选精排。

## 最终系统

```text
KuaiRec 行为与内容特征
  -> Behavior-aware RQ-KMeans
  -> 唯一 Semantic ID

用户历史
  -> Transformer 编码器
  -> GRU 自回归生成 Semantic ID
  -> Top-200 候选
  -> SASRec + 学习式精排
  -> Top-20
```

生成模型与 SASRec 均约 163 万参数，避免把模型容量差异误认为编码收益。

## 数据与结果

项目使用快手公开的 KuaiRec Big Matrix：

| 项目 | 数值 |
|---|---:|
| 正反馈行为 | 6,309,308 |
| 用户 | 7,174 |
| 视频 | 9,438 |
| Semantic ID 码本 | `[64, 64, 64]` |
| RQ 碰撞率 | 0 |
| 三级码本利用率 | 100% / 100% / 100% |

最终测试集结果：

| 阶段 | Recall@20 | NDCG@20 | Candidate Recall@200 |
|---|---:|---:|---:|
| 生成召回 | 0.196822 | 0.095620 | 0.561890 |
| 固定融合精排 | 0.213131 | 0.104453 | 0.561890 |
| 学习式精排 | **0.213688** | **0.104760** | 0.561890 |

完整诊断还验证了 Exact 与 Beam=500 的 Recall@200 完全相同，说明主要瓶颈不是
Beam 剪枝，而是第一级 Semantic ID 路由预测。失败的损失加权和蒸馏实验没有放入
最终运行目录，只在[实验结果](docs/results.md)中保留结论。

## 为什么属于生成式推荐

下一物品概率被分解为：

```text
P(item | history)
  = P(c1 | history)
  * P(c2 | history, c1)
  * P(c3 | history, c1, c2)
  * P(tail | history, c1, c2, c3)
```

解码器逐级生成 Token，并依赖先前已生成的 Token。推理时只能沿真实商品编码
构成的 Trie 扩展，因此不会生成目录外的无效商品。

## 核心工程能力

- 时间无泄漏的 leave-last-two-out 数据切分；
- 内容特征与 SASRec Item Embedding 融合；
- 带 PCA、白化、容量约束和碰撞消解的工业 RQ-KMeans；
- Transformer 历史编码与 GRU 自回归 Semantic ID 解码；
- 精确全目录评分与 Trie 约束 Beam Search；
- Semantic ID 召回、SASRec 精排和学习式融合；
- 等容量对照、验证集早停、checkpoint 恢复和 CUDA 混合精度；
- 索引 Schema、SHA-256 指纹、碰撞率和码本利用率检查。

## 目录

```text
configs/        可复现实验配置
data/           已处理数据；默认被 Git 忽略
docs/           代码导读、实验报告与复现材料
results/        正式指标摘要
scripts/        数据准备、建索引、训练、召回与精排入口
src/data/       数据加载、时间切分和训练样本
src/indexing/   Semantic ID 与 RQ-KMeans
src/models/     生成模型、SASRec 和候选精排器
src/training/   训练、评估、候选融合与学习式精排
tests/          索引、模型、数据和混合推荐测试
```

## 快速验证

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python scripts/run_demo.py --config configs/demo.json --output outputs/demo
```

Windows NVIDIA 环境见 [CUDA 运行指南](docs/windows-cuda.md)。

## KuaiRec 主流程

准备数据：

```bash
python scripts/prepare_kuairec.py \
  --source /path/to/KuaiRec.zip \
  --output data/kuairec_big
```

使用已有 SASRec checkpoint 构建 Behavior-aware RQ：

```bash
python scripts/build_behavior_rq.py \
  --config configs/kuairec_big_behavior_rq_cuda.json \
  --sasrec-checkpoint outputs/kuairec_big_cuda/sasrec_checkpoint.pt \
  --reference-codes outputs/kuairec_big_cuda/semantic_codes.npy \
  --output outputs/kuairec_big_behavior_rq/index
```

训练生成模型并进行多阶段评估：

```bash
python scripts/run_demo.py \
  --config configs/kuairec_big_behavior_rq_cuda.json \
  --output outputs/kuairec_big_behavior_rq/model

python scripts/run_hybrid.py \
  --config configs/kuairec_big_behavior_rq_cuda.json \
  --artifacts outputs/kuairec_big_behavior_rq/model \
  --sasrec-artifacts outputs/kuairec_big_cuda \
  --output outputs/kuairec_big_behavior_rq/model/hybrid_top200_metrics.json \
  --candidate-k 200 --beam-size 500 --device cuda

python scripts/run_learned_reranker.py \
  --config configs/kuairec_big_behavior_rq_cuda.json \
  --artifacts outputs/kuairec_big_behavior_rq/model \
  --sasrec-artifacts outputs/kuairec_big_cuda \
  --work-dir outputs/kuairec_big_behavior_rq/reranker \
  --candidate-k 200 --beam-size 500 --device cuda
```

## 文档

- [文档导航](docs/README.md)：推荐阅读顺序
- [项目演进](docs/project-evolution.md)：从 Demo 到正式系统的改进、实验与取舍
- [实验结果](docs/results.md)：正式实验、消融和失败实验结论
- [KuaiRec 实验](docs/kuairec-experiment.md)：数据协议与防泄漏设计
- [RQ-KMeans 设计](docs/rq-kmeans.md)：工业级 Semantic ID 索引
- [代码导读](docs/code-walkthrough.md)：核心模块与调用链

