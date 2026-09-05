# PCA 方差诊断增量包

将本增量包解压并覆盖到完整主线项目目录。它只读取现有 Sentence-T5 向量，
不会改动 `[64,64,64]` SID、检查点或训练结果。

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pca_variance_check.ps1
```

结果文件：

```text
data/kuairec_big_v2/pca_variance_comparison.json
```

该文件包含 PCA 128/192/256 维的累计方差解释率，以及达到 85%/90%/95%
所需的最小维数。判断规则：PCA-128 达到 85% 就保留当前设置；否则优先选择
达到 85% 的最小候选维数。运行完成后，把 JSON 文件发回即可。
