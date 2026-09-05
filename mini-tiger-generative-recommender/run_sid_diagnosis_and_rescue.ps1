$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & ".\.venv\Scripts\Activate.ps1"
}

$data = "data/kuairec_big_v2"
$oldSid = "$data/rq_64_64_64/semantic_codes.npy"
$newDirectory = "$data/rq_128_64_32"
$newSid = "$newDirectory/semantic_codes.npy"
$embeddings = "$data/sentence_t5_embeddings.npy"

python scripts/check_cuda.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts/analyze_sid_semantics.py `
  --codes $oldSid `
  --item-ids "$data/item_ids.npy" `
  --item-texts "$data/item_texts.csv" `
  --codebook-sizes 64 64 64 `
  --output "$data/sid_semantics_64_64_64.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not (Test-Path $newSid)) {
    python scripts/build_sentence_rq_kmeans.py `
      --embeddings $embeddings `
      --output $newDirectory `
      --codebook-sizes 128 64 32 `
      --pca-dim 128 `
      --seed 2026
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

python scripts/analyze_sid_semantics.py `
  --codes $newSid `
  --item-ids "$data/item_ids.npy" `
  --item-texts "$data/item_texts.csv" `
  --codebook-sizes 128 64 32 `
  --output "$data/sid_semantics_128_64_32.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts/run_generative_v2.py `
  --config configs/kuairec_big_v2_rescue_128_64_32_h256_l4_quick10.json `
  --output outputs/kuairec_big_main_rescue_128_64_32_h256_l4_quick10
exit $LASTEXITCODE
