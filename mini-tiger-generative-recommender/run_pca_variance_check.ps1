$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & ".\.venv\Scripts\Activate.ps1"
}

$embeddingPath = "data/kuairec_big_v2/sentence_t5_embeddings.npy"
$outputPath = "data/kuairec_big_v2/pca_variance_comparison.json"

if (-not (Test-Path $embeddingPath)) {
    throw "Missing $embeddingPath. Generate the Sentence-T5 embeddings first."
}

python scripts/analyze_pca_variance.py `
  --embeddings $embeddingPath `
  --dims 128 192 256 `
  --thresholds 0.85 0.90 0.95 `
  --output $outputPath

if ($LASTEXITCODE -eq 0) {
    Write-Host "Finished: $outputPath"
}
exit $LASTEXITCODE
