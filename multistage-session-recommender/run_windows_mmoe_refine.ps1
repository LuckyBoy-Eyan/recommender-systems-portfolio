$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE'); assert torch.cuda.is_available(), '请先安装支持 CUDA 的 PyTorch'"

python scripts/train_multitask_ranker.py `
  --features outputs/ranker_datasets/features `
  --output outputs/windows_mmoe_refined_validation `
  --model mmoe --epochs 10 --patience 2 --min-delta 0.00005 `
  --batch-size 4096 --device cuda

Write-Host "MMoE微调已完成；请压缩 outputs/windows_mmoe_refined_validation"
