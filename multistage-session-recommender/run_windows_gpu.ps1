$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE'); assert torch.cuda.is_available(), '请先安装支持 CUDA 的 PyTorch'"

python scripts/run_two_tower_recall.py `
  --processed data/processed/retailrocket `
  --output outputs/windows_two_tower_validation `
  --epochs 10 `
  --batch-size 512 `
  --embedding-dim 64 `
  --topk 50 `
  --device cuda `
  --early-stopping `
  --min-epochs 3 `
  --patience 2 `
  --min-delta 0.0002

Write-Host "完成。指标：outputs/windows_two_tower_validation/metrics.json"
