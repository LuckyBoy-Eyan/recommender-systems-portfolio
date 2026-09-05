# 内容 Item Embedding

最终主线不使用 SASRec Item Embedding，也不把用户行为统计融合进 SID。物品表示完全来自
内容侧，行为信号由后续用户序列 Transformer 建模。

## Sentence-T5

每个视频将标题与多级类目整理为一条文本，例如：

```text
title: 篮球精彩集锦; category_1: 体育; category_2: 篮球
```

使用 `sentence-transformers/sentence-t5-base` 编码为 768 维稠密向量，并执行 L2 归一化。

## PCA 与量化

Sentence-T5 768 维向量通过 PCA 降至 128 维，不使用 whitening，随后再次 L2 归一化。
在该空间依次运行容量为 `[128,64,32]` 的残差 K-Means。每一级量化前一级残差，得到
`<c1,c2,c3>`；同前缀物品追加局部 tail 编号，形成唯一 `<c1,c2,c3,tail>`。

9,438 个物品得到 5,218 个三级前缀；4,220 个额外碰撞物品需要 tail，完整 SID 碰撞率为
0。Tail 没有跨碰撞组语义，只承担唯一标识作用。
