# Windows Two-Tower V2

本包只使用训练集和验证集，不读取测试集。默认适配 RTX 5060 8GB：FP16、Batch 1024，
若出现 CUDA OOM，把 PowerShell 文件中的 `--batch-size 1024` 改为 `512` 后重新运行。

1. 安装 Python 3.11/3.12 和 CUDA 版 PyTorch。
2. 执行 `pip install -r requirements.txt`。
3. 执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_windows_two_tower_v2.ps1
```

首次运行会生成并缓存 414k×96 的去重困难负样本，随后训练双塔。最多15轮，最少5轮，
Patience=4；Item2Vec前2轮冻结。完成后回传：

`outputs/windows_two_tower_v2_results.zip`

其中包含模型、训练指标、Top300验证候选和困难负样本缓存。训练过程不评估测试集。
