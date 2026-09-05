# SID语义诊断与快速补救实验

把增量包覆盖到现有 MiniTIGER Windows 主线目录。本实验使用独立的 SID、配置、
检查点和结果目录，不修改 `[64,64,64]` 正式结果。

执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_sid_diagnosis_and_rescue.ps1
```

流程依次为：

1. 诊断现有 `[64,64,64]` 的逐层利用率、联合空间占用率、碰撞率；
2. 按一级/二级/三级类目计算同类与异类 SID 汉明距离、共享前缀率和一级 token NMI；
3. 从现有 Sentence-T5 向量构建 `[128,64,32]` SID，并执行同样诊断；
4. 训练 Hidden=256、Encoder/Decoder 各4层的10轮快速实验，以 HR@200 监控。

主要输出：

```text
data/kuairec_big_v2/sid_semantics_64_64_64.json
data/kuairec_big_v2/sid_semantics_128_64_32.json
outputs/kuairec_big_main_rescue_128_64_32_h256_l4_quick10/metrics.json
```

逐层利用率与联合空间占用率含义不同：只有9,438个物品时，不可能填满262,144个
三级组合；判断码本坍缩应看每一级 token 利用率、熵和桶大小。

注意：10轮实验用于快速判断训练趋势，不能代替与原模型相同早停协议的正式实验。
