# 后续完善方向

## 已完成

- 按时间顺序切分完整 Session，并构建无泄漏的 Leave-Last-Event-Out 标签
- 实现 Recent、Popularity、ItemCF 和 Item2Vec 四路召回；Co-visitation 经消融后删除
- 正式排序链路只保留Shared-Bottom，并使用任务Mask处理三目标监督
- 支持候选集合 Recall 和分召回源诊断
- 构建候选、物品、Session、转化率、交叉特征和时间新近性特征
- 训练 clicks/carts/orders 分目标 Hard Negative 排序模型
- 支持 Weighted Recall@20 离线评估
- 提供 OTTO 风格真实数据加载器、测试和实验结果文档

## 真实数据扩展

1. 固定一个 Kaggle OTTO 子集进行基准实验，并公开硬件配置与运行耗时。
2. 加入 LightGBM LambdaRank，并与当前依赖较轻的 sklearn baseline 对比。
3. 加入 skip-gram Item2Vec 和 ANN 向量检索服务。
4. 报告各召回源的独占命中情况，并做候选数量消融实验。
5. 在现有日级 point-in-time 快照基础上增加增量物化，并完善线上/线下一致性检查。

