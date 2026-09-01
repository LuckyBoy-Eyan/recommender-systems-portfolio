$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python scripts/evaluate_ranker_baseline.py `
  --features outputs/ranker_datasets/features/validation.parquet `
  --output outputs/windows_rrf_baseline_validation/metrics.json `
  --topk 20

Compress-Archive `
  -Path outputs/windows_rrf_baseline_validation `
  -DestinationPath outputs/windows_rrf_baseline_validation.zip `
  -Force

Write-Host "RRF验证基线已完成：outputs/windows_rrf_baseline_validation.zip"
