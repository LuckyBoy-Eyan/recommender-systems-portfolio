$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -c "import torch; print('PyTorch:',torch.__version__); print('GPU:',torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"

python scripts/train_multitask_ranker.py --features outputs/ranker_datasets_v2/features --output outputs/windows_mmoe_v3_validation --model mmoe --epochs 15 --patience 4 --min-delta 0.0002 --batch-size 4096 --batch-rows 200000 --learning-rate 0.001 --negatives-per-epoch 160 --average-candidates 783.2108685 --device cuda
python scripts/train_multitask_ranker.py --features outputs/ranker_datasets_v2/features --output outputs/windows_ple_v3_validation --model ple --epochs 15 --patience 4 --min-delta 0.0002 --batch-size 4096 --batch-rows 200000 --learning-rate 0.001 --negatives-per-epoch 160 --average-candidates 783.2108685 --device cuda

Compress-Archive -Path outputs/windows_mmoe_v3_validation,outputs/windows_ple_v3_validation -DestinationPath outputs/windows_rankers_v3_results.zip -Force
Get-FileHash outputs/windows_rankers_v3_results.zip -Algorithm SHA256
