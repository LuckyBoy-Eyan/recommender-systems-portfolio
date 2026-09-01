$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE'); assert torch.cuda.is_available(), '请安装CUDA版PyTorch'"

python scripts/run_two_tower_v2_windows.py `
  --processed data/processed/retailrocket `
  --output outputs/windows_two_tower_v2_validation `
  --epochs 15 `
  --batch-size 1024 `
  --topk 300

Compress-Archive -Path outputs/windows_two_tower_v2_validation -DestinationPath outputs/windows_two_tower_v2_results.zip -Force
Write-Host "完成：outputs/windows_two_tower_v2_results.zip"
