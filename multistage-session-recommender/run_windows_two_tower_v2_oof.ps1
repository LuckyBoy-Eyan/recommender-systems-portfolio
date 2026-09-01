$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE'); assert torch.cuda.is_available(), '请安装CUDA版PyTorch'"

foreach ($fold in 0..3) {
  Write-Host "========== OOF Fold $fold / 3 =========="
  python scripts/run_two_tower_v2_windows.py `
    --processed data/processed/retailrocket `
    --output "outputs/windows_two_tower_v2_oof/fold_$fold" `
    --oof-fold $fold `
    --oof-folds 4 `
    --warmup-fraction 0.2 `
    --epochs 6 `
    --batch-size 1024 `
    --topk 300 `
    --device cuda
}

Compress-Archive -Path outputs/windows_two_tower_v2_oof -DestinationPath outputs/windows_two_tower_v2_oof_results.zip -Force
Get-FileHash outputs/windows_two_tower_v2_oof_results.zip -Algorithm SHA256
Write-Host "完成：outputs/windows_two_tower_v2_oof_results.zip"
