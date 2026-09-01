$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$out = "outputs/final_test_frozen"
if (Test-Path "$out/FINAL_TEST_COMPLETE.json") { throw "最终测试已经完成，禁止重复运行或据此继续调参。" }
$required = @(
  "data/processed/retailrocket/frozen_recall_indexes.joblib",
  "data/processed/retailrocket/frozen_item2vec_embeddings_v2.npz",
  "data/processed/retailrocket/test_samples.parquet",
  "outputs/final_frozen_models/two_tower_v2.pt",
  "outputs/final_frozen_models/ple.pt"
)
foreach ($path in $required) {
  if (-not (Test-Path $path)) { throw "缺少最终测试依赖文件: $path" }
}
New-Item -ItemType Directory -Force -Path $out | Out-Null
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
if (Test-Path "$out/two_tower_candidates.parquet") {
  python -c "import pandas as pd; p='$out/two_tower_candidates.parquet'; x=pd.read_parquet(p,columns=['session']); assert len(x)==7710000 and x.session.nunique()==25700, (len(x),x.session.nunique()); print('复用已校验的双塔测试 Top300:',len(x))"
} else {
  python scripts/score_two_tower_v2_train_windows.py --processed data/processed/retailrocket --samples-file test_samples.parquet --group-by session --checkpoint outputs/final_frozen_models/two_tower_v2.pt --output "$out/two_tower_candidates.parquet" --topk 300 --batch-size 1024 --device cuda
}
python scripts/build_frozen_test_ranker_dataset.py --processed data/processed/retailrocket --tower "$out/two_tower_candidates.parquet" --output $out
python scripts/score_multitask_checkpoint.py --checkpoint outputs/final_frozen_models/ple.pt --features "$out/test_features.parquet" --output "$out/ple_scores.parquet" --model ple --device cuda --batch-rows 200000
foreach ($k in 5,10,20,50) {
  python scripts/evaluate_unified_scores.py --scores "$out/ple_scores.parquet" --score-column final_score --processed data/processed/retailrocket --samples-file test_samples.parquet --output "$out/ple_metrics_at_$k.json" --topk $k
  python scripts/evaluate_unified_scores.py --scores "$out/test_features.parquet" --score-column rrf_score --processed data/processed/retailrocket --samples-file test_samples.parquet --output "$out/rrf_metrics_at_$k.json" --topk $k
}
'{"test_evaluated":true,"model":"ple_v3_epoch9","fusion_probabilities":[0.05,0.30,0.40],"rrf_role":"independent_baseline_only"}' | Set-Content "$out/FINAL_TEST_COMPLETE.json"
Compress-Archive -Path "$out/ple_scores.parquet","$out/*metrics*.json","$out/FINAL_TEST_COMPLETE.json" -DestinationPath outputs/windows_final_frozen_test_results.zip -Force
Get-FileHash outputs/windows_final_frozen_test_results.zip -Algorithm SHA256
