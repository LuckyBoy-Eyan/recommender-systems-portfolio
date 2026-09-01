# 冻结最终测试（只运行一次）

解压覆盖到原项目根目录，确认`data/processed/retailrocket/test_samples.parquet`存在，然后执行：

`powershell -ExecutionPolicy Bypass -File .\run_windows_final_frozen_test.ps1`

配置已经冻结：双塔V2第6轮、PLE V3第9轮、三塔概率权重`0.05/0.30/0.40`。RRF仅单独作为基线。脚本不含调参或训练逻辑；完成后会写入防重复标记。返回`outputs/windows_final_frozen_test_results.zip`。
