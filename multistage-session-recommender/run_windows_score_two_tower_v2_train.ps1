$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
python scripts/score_two_tower_v2_train_windows.py `
  --checkpoint outputs/windows_two_tower_v2_validation/two_tower_v2.pt `
  --output outputs/windows_two_tower_v2_train_candidates/two_tower_candidates.parquet `
  --topk 300 --batch-size 1024 --device cuda
Compress-Archive -Path outputs/windows_two_tower_v2_train_candidates -DestinationPath outputs/windows_two_tower_v2_train_candidates.zip -Force
Get-FileHash outputs/windows_two_tower_v2_train_candidates.zip -Algorithm SHA256
