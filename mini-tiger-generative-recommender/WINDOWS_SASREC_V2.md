# Windows RTX 5060：SASRec V2 全量重训

这个包只用于公平 baseline，不需要 Sentence-T5、RQ-KMeans 或 Semantic ID。

## 数据协议

- 完整保留 12,487,057 条正负事件；
- 正反馈：watch ratio >= 0.7 且播放时长 >= 5 秒；
- 输入加入 Feedback Type Embedding；
- 最后两个正事件分别作为验证和测试目标；
- 负目标与 padding 标签均为 `-100`；
- 推理时第一版不过滤历史物品。

## 环境

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-windows.txt
```

随后按照 PyTorch 官方 Windows 安装选择器安装支持 RTX 50 系列的 CUDA 版本。

## 检查与训练

```powershell
python scripts/check_cuda.py `
  --config configs/kuairec_big_sasrec_v2_cuda.json

python -m pytest -q

python scripts/run_masked_sasrec_v2.py `
  --config configs/kuairec_big_sasrec_v2_cuda.json `
  --output outputs/kuairec_big_sasrec_v2_cuda
```

每轮原子保存 `masked_sasrec_checkpoint.pt`，中断后执行同一命令会继续训练。
最终指标写入 `outputs/kuairec_big_sasrec_v2_cuda/metrics.json`。

训练最多100轮，前2轮线性 warmup，随后 cosine decay 到 `1e-5`；验证
HitRate@20 连续10轮没有改善时提前停止。训练只为正目标位置计算全目录 logits，
负事件仍完整经过 Transformer，因此不会改变 masked-loss 协议。
