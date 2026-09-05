# SID码本 A/B 增量实验

先解压完整主线包，再把本增量包覆盖到同一目录。确保已经生成：

```text
data/kuairec_big_v2/sentence_t5_embeddings.npy
```

## 1. 构建 `[32,32,32]`

```powershell
python scripts/build_sentence_rq_kmeans.py `
  --embeddings data/kuairec_big_v2/sentence_t5_embeddings.npy `
  --output data/kuairec_big_v2/rq_32_32_32 `
  --codebook-sizes 32 32 32 --pca-dim 128
```

## 2. 构建 `[64,32,32]`

```powershell
python scripts/build_sentence_rq_kmeans.py `
  --embeddings data/kuairec_big_v2/sentence_t5_embeddings.npy `
  --output data/kuairec_big_v2/rq_64_32_32 `
  --codebook-sizes 64 32 32 --pca-dim 128
```

## 3. 自动比较

```powershell
python scripts/compare_rq_manifests.py `
  --sid32 data/kuairec_big_v2/rq_32_32_32/manifest.json `
  --sid64 data/kuairec_big_v2/rq_64_32_32/manifest.json `
  --output data/kuairec_big_v2/sid_ab_comparison.json
```

先把 `sid_ab_comparison.json` 发回分析。若仍需模型指标，用完全相同配置分别训练：

```powershell
python scripts/run_generative_v2.py `
  --config configs/kuairec_big_v2_cuda.json `
  --semantic-codes data/kuairec_big_v2/rq_32_32_32/semantic_codes.npy `
  --output outputs/main_sid_32_32_32

python scripts/run_generative_v2.py `
  --config configs/kuairec_big_v2_cuda.json `
  --semantic-codes data/kuairec_big_v2/rq_64_32_32/semantic_codes.npy `
  --output outputs/main_sid_64_32_32
```

两个输出目录必须分开，避免检查点互相覆盖。
