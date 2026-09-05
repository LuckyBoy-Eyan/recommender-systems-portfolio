$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & ".\.venv\Scripts\Activate.ps1"
}

python scripts/check_cuda.py `
  --config configs/kuairec_big_sasrec_v2_fair100_hr200_cuda.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts/run_masked_sasrec_v2.py `
  --config configs/kuairec_big_sasrec_v2_fair100_hr200_cuda.json `
  --output outputs/kuairec_big_sasrec_v2_fair100_hr200_cuda
exit $LASTEXITCODE
