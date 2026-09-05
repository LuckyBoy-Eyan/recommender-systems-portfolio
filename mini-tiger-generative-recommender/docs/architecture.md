# 当前架构

## 目标与边界

当前系统只评估生成式召回：输入完整正负反馈历史，生成下一条正反馈物品的 Semantic ID，
再映射为 Top-K 物品列表。系统不包含第二阶段精排；SASRec 仅作为独立公平基线。

## 数据流

```text
标题/类目 -> Sentence-T5 768D -> PCA 128D -> RQ-KMeans [128,64,32]
                                              -> collision tail -> 唯一 SID

历史 SID + Feedback Type + Position -> Transformer Encoder (6×H256)
目标 SID 右移前缀             -> Causal Decoder (6×H256)
                                -> c1 -> c2 -> c3 -> tail
                                -> Trie Beam Search -> Top-K item
```

历史物品表示由各 SID 层 Embedding、反馈类型 Embedding与位置 Embedding 相加得到。负反馈
在输入侧可见，但不作为监督目标；正目标损失仍可通过注意力间接更新负反馈上下文表示。

Decoder 训练时使用 teacher forcing，推理从 start token 开始。Trie 只允许扩展目录中存在的
前缀，完整四段 SID 必须映射到唯一真实物品。

## 最终配置

- Hidden 256，Encoder 6 层，Decoder 6 层，4 Heads，FFN 1024；
- 最大历史长度 100，Dropout 0.1；
- 训练期验证 Beam 200，补评 Beam 500；
- FP16、fused AdamW、cosine 学习率与断点续训。
