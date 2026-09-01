# Windows GTX 5060 运行说明

此包用于在 Windows CUDA GPU 上训练双塔，不读取测试集。

1. 安装 Python 3.11 或 3.12。
2. 根据 PyTorch 官网安装与你驱动匹配的 CUDA 版 PyTorch。
3. 安装其余依赖：`pip install -r requirements.txt`。
4. 在 PowerShell 中运行：`powershell -ExecutionPolicy Bypass -File .\run_windows_gpu.ps1`。

脚本首先检查 CUDA；若 PyTorch 只能看到 CPU，会立即停止。GTX 5060 8GB 默认使用
FP16 混合精度和 batch 512。最多训练10轮，每轮验证，至少训练3轮；连续2轮综合召回
无改善时停止并恢复最佳轮次。若显存不足，将脚本中的 batch 改为 256。

完成后把以下目录压缩传回 Mac：

- `outputs/windows_two_tower_validation/metrics.json`
- `outputs/windows_two_tower_validation/two_tower.pt`
- `outputs/windows_two_tower_validation/two_tower_candidates.parquet`

当前包不运行最终测试，也暂不运行全量滚动 OOF；Mac 端会先完成候选分块压缩，之后再
提供 PLE/MMoE GPU 训练任务。
