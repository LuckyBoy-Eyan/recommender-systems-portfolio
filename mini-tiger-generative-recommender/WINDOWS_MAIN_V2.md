# Windows RTX 5060：MiniTIGER V2 主线

## 协议

- 完整正负事件作为上下文；
- Feedback Type Embedding 显式区分正负；
- 最后两个正事件作为验证、测试目标；
- 只对正目标训练；
- Sentence-T5 768维内容向量；
- RQ-KMeans `[32,32,32]`，追加 tail 保证唯一；
- Transformer Encoder-Decoder + Trie Beam Search 200；
- 第一版不过滤历史物品。

## 1. 环境

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple `
  -r requirements-main-windows.txt
```

随后通过 PyTorch 官方 Windows 安装选择器安装支持 RTX 50 系列的 CUDA wheel。

## 2. Sentence-T5

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
python scripts/build_sentence_t5_embeddings.py `
  --texts data/kuairec_big_v2/item_texts.csv `
  --output data/kuairec_big_v2/sentence_t5_embeddings.npy `
  --model sentence-transformers/sentence-t5-base `
  --batch-size 64 --device cuda
```

## 3. RQ-KMeans SID

```powershell
python scripts/build_sentence_rq_kmeans.py `
  --embeddings data/kuairec_big_v2/sentence_t5_embeddings.npy `
  --output data/kuairec_big_v2/rq_kmeans `
  --codebook-size 32 --levels 3 --pca-dim 128
```

确认 `manifest.json` 中 `complete_collision_rate` 为0。

## 4. 主模型训练

```powershell
python scripts/run_generative_v2.py `
  --config configs/kuairec_big_v2_cuda.json `
  --output outputs/kuairec_big_main_v2_cuda
```

最多100轮，2轮 warmup 后 cosine decay 到 `1e-5`，Validation HitRate@20
连续10轮不提升则早停。检查点包含模型、优化器、AMP scaler 和 scheduler，执行
同一命令即可续训。最终同时输出 Beam Top-K 指标和 Exact AUC/UAUC。
