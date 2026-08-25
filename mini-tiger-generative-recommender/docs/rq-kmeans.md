# 工业 RQ-KMeans 索引

## 它替换了什么

旧方案在父前缀内反复做树式 KMeans，能制造唯一编号，但它更接近“分层目录”
而不是标准残差量化。现在主配置改为全局多级 RQ-KMeans：

```text
原始 item feature
  -> PCA / whitening / L2 normalize
  -> KMeans(residual_0) -> token_0
  -> KMeans(residual_1) -> token_1
  -> KMeans(residual_2) -> token_2
  -> 最后一级最小代价去碰撞
  -> 极端情况 Tail Token 兜底
```

每一级的输入是上一层尚未表达的残差：

```text
residual_(l+1) = residual_l - codebook_l[token_l]
```

因此完整编码不是人为切开的树路径，而是多个码字共同重构物品向量。

## 工业化能力

- `rq_backend=auto`：安装 Faiss 时用 Faiss K-Means，否则用 sklearn；
- PCA、白化和归一化参数随索引保存，不依赖线上重新拟合；
- `rq_max_balance_ratio` 限制热点 token 的最大容量；
- 最后一级对碰撞前缀执行 Hungarian 最小代价匹配；
- 每层记录量化误差、残差范数、使用 token 数和最大/最小桶；
- 索引包含 Schema 版本和 SHA-256 指纹，便于一致性检查与回滚；
- 旧 `hierarchical` 和 `residual` 方法仍可作为消融实验运行。

## 配置参数

| 参数 | 含义 | KuaiRec Big |
|---|---|---:|
| `codebook_sizes` | 每一级 token 词表大小 | `[64,64,64]` |
| `rq_backend` | `auto` / `faiss` / `sklearn` | `auto` |
| `rq_pca_dim` | PCA 输出维度；`null` 表示不做 PCA | `96` |
| `rq_whiten` | 是否按主成分方差白化 | `true` |
| `rq_l2_normalize` | 聚类前是否做行 L2 归一化 | `true` |
| `rq_niter` | 每次 K-Means 最大迭代轮数 | `25` |
| `rq_nredo` | 多次初始化次数 | `3` |
| `rq_use_gpu` | Faiss 是否使用 CUDA GPU | `false` |
| `rq_max_balance_ratio` | 最大桶相对平均桶的倍数 | `1.25` |
| `rq_resolve_collisions` | 是否做末级最小代价去碰撞 | `true` |

`rq_use_gpu` 只控制 Faiss；Apple MPS 不属于 Faiss CUDA GPU。需要 GPU 建索引时，
应在带 CUDA 的 Linux 环境安装对应 Faiss 版本并显式设置为 `true`。

## 发布产物

运行 `scripts/run_demo.py` 后，输出目录中新增：

- `rq_kmeans_index.npz`：物品编码、PCA 数组、各层 centroid；
- `rq_kmeans_manifest.json`：版本、指纹、后端、配置和质量诊断；
- `semantic_codes.npy`：包含最终 Tail 层的推荐模型目录编码；
- `metrics.json`：推荐效果、训练曲线和索引摘要。

线上部署时应把 manifest 指纹同时写入模型版本；只有模型指纹和索引指纹一致时
才允许加载，避免“模型按旧 token 预测、服务按新 token 查商品”。
`load_rq_kmeans_artifact` 和 `encode_with_rq_kmeans` 可用同一套预处理与码本编码
新物品；增量物品如果撞码，应追加 Tail 或触发目录重建，不能静默覆盖旧映射。

## 验证

```bash
python -m pytest -q

python scripts/run_demo.py \
  --config configs/kuairec_smoke.json \
  --output outputs/kuairec_smoke_rq
```

当前测试覆盖确定性、token 范围、残差误差、容量上限、碰撞消解以及发布产物。
旧树式 KMeans 实现仍保留在 `src/indexing/semantic_ids.py`，但失败的旧实验配置和
临时输出已从正式目录移除。

