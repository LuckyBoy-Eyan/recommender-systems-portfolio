$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & ".\.venv\Scripts\Activate.ps1"
}

$embeddingPath = "data/kuairec_big_v2/sentence_t5_embeddings.npy"
$sidDirectory = "data/kuairec_big_v2/rq_64_64_64"
$sidPath = "$sidDirectory/semantic_codes.npy"
$manifestPath = "$sidDirectory/manifest.json"
$configPath = "configs/kuairec_big_v2_cuda_64_64_64.json"
$outputPath = "outputs/kuairec_big_main_v2_cuda_64_64_64"

if (-not (Test-Path $embeddingPath)) {
    throw "Missing $embeddingPath. Generate the Sentence-T5 embeddings first."
}

python scripts/check_cuda.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ((-not (Test-Path $sidPath)) -or (-not (Test-Path $manifestPath))) {
    python scripts/build_sentence_rq_kmeans.py `
      --embeddings $embeddingPath `
      --output $sidDirectory `
      --codebook-sizes 64 64 64 `
      --pca-dim 128 `
      --seed 2026
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
if (($manifest.codebook_sizes -join ",") -ne "64,64,64") {
    throw "The existing SID manifest is not a 64,64,64 codebook."
}
if ([double]$manifest.complete_collision_rate -ne 0.0) {
    throw "The generated Semantic IDs are not unique."
}

python scripts/run_generative_v2.py `
  --config $configPath `
  --output $outputPath
exit $LASTEXITCODE
