# 可复现实验指标

当前正式数据口径为：

```text
Top3000 商品
20,000 个 Session
96,516 条行为
2,802 个实际出现商品
14,000 / 3,000 / 3,000 训练、验证、测试 Session
```

本目录只保留适合版本管理的轻量结果：

- `retailrocket_top3000_session20000_validation`：锁定方案前的验证实验；
- `retailrocket_top3000_session20000_final`：锁定配置后的唯一测试评估。
- `topk_per_source_ablation`：候选深度消融的指标与分析；
- `warmup_ablation`：不同预热窗口的指标；
- 根目录 JSON：特征遮蔽与共享底座容量实验。

仓库保留 `metrics.json`、`manifest.json` 和汇总分析 JSON，便于核验报告中的
数值。候选明细、训练样本、排序器文件和逐快照审计 CSV 均可由脚本重新生成，
体积较大且已被 Git 忽略，不纳入作品集。
