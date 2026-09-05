# MiniTIGER：基于 Semantic ID 的生成式召回

MiniTIGER 在 KuaiRec Big Matrix 上探索一个问题：只使用内容语义构建物品 Semantic ID，
再通过 Transformer 自回归生成下一个正反馈物品，能否形成有效的端到端生成式召回系统。
当前公开版本只包含召回，不包含精排或融合模型。

## 系统结构

```text
视频标题与类目
      ↓
Sentence-T5（768 维）
      ↓
PCA（128 维，无白化）+ L2 Normalize
      ↓
RQ-KMeans [128, 64, 32] + collision tail
      ↓
唯一 Semantic ID <c1,c2,c3,tail>

完整正负反馈历史 + Feedback Type Embedding
      ↓
6 层 Transformer Encoder（Hidden=256）
      ↓
6 层 Causal Transformer Decoder
      ↓
Trie 约束 Beam Search
      ↓
Top-K 生成式召回列表
```

SASRec 是独立公平基线，不参与 Semantic ID 构建、MiniTIGER 训练或候选排序。

## 数据协议

- 数据集：KuaiRec Big Matrix，7,174 用户、9,438 视频、12,487,057 条交互；
- 正反馈：观看比例 `>= 0.7` 且播放时长 `>= 5s`；其余有效交互记为负反馈；
- 输入保留完整正负序列，并显式加入正/负 Feedback Type Embedding；
- 负反馈只作为上下文，不作为预测目标；模型只学习生成深度观看物品；
- 每位用户倒数第二个正反馈作为验证目标，最后一个正反馈作为测试目标；
- 推理时不屏蔽历史物品，避免错误删除重复观看目标。

## Semantic ID

最终码表为 `[128,64,32]`，三级前缀理论容量为 262,144。9,438 个物品形成 5,218 个
不同前缀；4,220 个额外碰撞物品通过局部 `tail` 编号消歧，完整 SID 碰撞率为 0。
前三层负责语义结构，tail 只负责同前缀物品的唯一标识。

## 公平实验

Fair100 协议将 MiniTIGER 与 SASRec 的训练规模统一为 716,169 个正目标，并统一使用
验证集 HR@200 早停。MiniTIGER 最终模型为 H256、Encoder 6 层、Decoder 6 层、4 Heads、
FFN 1024，最大历史长度 100；训练支持 FP16、fused AdamW 和断点续训。

### 测试集结果

| 模型 | HR@20 | HR@200 | HR@500 | NDCG@20 | MRR@20 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| SASRec V2 Fair100 | **0.2280** | **0.6140** | **0.8372** | 0.1101 | 0.0769 | **0.9696** |
| MiniTIGER L6 | 0.2234 | 0.5891 | 0.8032 | **0.1150** | **0.0848** | 0.9628 |

结果表明：SASRec 的候选覆盖更高，MiniTIGER 的头部 NDCG/MRR 更高。项目保留这一真实
结论，不把生成式模型描述为全面优于基线。测试集已用于阶段性分析，后续模型选择只应使用
验证集。

## 运行方式

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

数据与主实验入口：

```bash
python scripts/prepare_kuairec_v2.py \
  --source data/raw/KuaiRec.zip \
  --output data/kuairec_big_v2

python scripts/build_sentence_t5_embeddings.py \
  --texts data/kuairec_big_v2/item_texts.csv \
  --output data/kuairec_big_v2/sentence_t5_embeddings.npy

python scripts/build_sentence_rq_kmeans.py \
  --embeddings data/kuairec_big_v2/sentence_t5_embeddings.npy \
  --output data/kuairec_big_v2/rq_128_64_32 \
  --codebook-sizes 128 64 32 \
  --pca-dim 128

python scripts/run_generative_v2.py \
  --config configs/kuairec_big_v2_128_64_32_h256_l6_full100.json \
  --output outputs/kuairec_big_v2_l6

python scripts/run_masked_sasrec_v2.py \
  --config configs/kuairec_big_sasrec_v2_fair100_hr200_cuda.json \
  --output outputs/kuairec_big_sasrec_v2_fair100
```

原始数据、Sentence-T5 权重、训练检查点和大体积中间产物不提交至 Git。

## 目录

```text
configs/          CPU、CUDA 与公平实验配置
scripts/          数据准备、SID 构建、训练和诊断入口
src/indexing/     RQ-KMeans 与 Semantic ID
src/models/       MiniTIGER 与独立 SASRec 基线
src/training/     训练、Exact 与 Trie Beam 评估
docs/             架构、数据协议和实验说明
tests/            单元与回归测试
```

详见[全量实验报告](docs/experiment-report.md)和[文档导航](docs/README.md)。
