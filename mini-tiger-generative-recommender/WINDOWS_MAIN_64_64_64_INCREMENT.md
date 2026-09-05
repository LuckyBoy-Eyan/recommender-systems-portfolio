# MiniTIGER `[64,64,64]` 主线增量包

把本压缩包解压并覆盖到之前的完整主线包目录。它不会覆盖 `[32,32,32]`
配置或结果，而是使用独立的 SID 和训练输出目录。

## 运行

确认之前已经生成：

```text
data/kuairec_big_v2/sentence_t5_embeddings.npy
```

然后在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_main_64_64_64.ps1
```

脚本会依次：

1. 检查 CUDA；
2. 生成或复用 `[64,64,64]` RQ-KMeans SID；
3. 检查完整 SID 冲突率必须为 0；
4. 在独立目录中训练主模型；
5. 完成 Beam 指标和 Exact AUC/UAUC 评估。

## 输出

```text
data/kuairec_big_v2/rq_64_64_64/manifest.json
outputs/kuairec_big_main_v2_cuda_64_64_64/semantic_checkpoint.pt
outputs/kuairec_big_main_v2_cuda_64_64_64/metrics.json
```

训练中断后再次运行同一个命令，会从独立检查点继续训练。最多 100 轮，前 2 轮
warmup，之后 cosine 衰减到 `1e-5`；Validation HitRate@20 连续 10 轮没有提升时
提前停止。
