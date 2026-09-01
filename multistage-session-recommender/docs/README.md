# 文档导航

建议按以下顺序阅读：

1. [完整实验报告](complete-experiment-report.md)：数据、七路召回、双塔、PLE、融合与最终测试；
2. [结果摘要](results.md)：冻结验证与一次性测试核心指标；
3. [面试指南](interview-guide.md)：简历描述、三分钟讲稿和常见追问；
4. [项目演进](project-evolution.md)：从 Top3000 原型升级到全量目录七路召回；
5. [代码导读](code-walkthrough.md)：核心模块和调用链；
6. [历史实验归档](experiments/README.md)：旧口径候选深度、Item2Vec、排序器和预热实验；
7. [后续研究方向](roadmap.md)。

## 结果口径

- 当前正式结果：23.5 万事件商品、七路召回、PLE V3、一次性冻结测试；
- 历史 Top3000、四路召回和 Shared-Bottom 指标只作为早期演进记录；
- 当前文档统一使用 HitRate@K；旧代码中的单目标 Recall@K 与其数值等价；
- RRF 是独立强基线，不用于截断 PLE 候选。
