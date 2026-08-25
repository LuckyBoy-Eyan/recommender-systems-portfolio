# 面试讲稿与追问

## 简历描述

基于 RetailRocket 构建多阶段会话推荐系统：先按 30 分钟无操作间隔切分 Session，
再保留 Top3000 商品和 20,000 个合格 Session；通过 Recent、Popularity、ItemCF、
Item2Vec 四路召回生成候选，构造 Point-in-Time 特征与 Hard Negative，使用
Shared-Bottom 多任务排序。最终测试 Candidate Recall 为 `0.9310`，行为加权候选
上限为 `0.9802`；Shared-Bottom Weighted Recall@20 为 `0.9668`，相对 RRF 基线
`0.7770` 绝对提升 `0.1898`。

## 三分钟讲解

这个项目解决的是电商会话内的下一物品推荐。原始 RetailRocket 数据只有时间、访客、
事件类型和商品字段，我先把 view、addtocart、transaction 映射为 clicks、carts、
orders。在完整用户行为流上，当前事件与上一事件间隔超过 30 分钟时开启新 Session；
随后筛选 Top3000 商品，删除过滤后少于三次行为的 Session。若 Session 最大时间戳并列，
无法唯一确定下一事件标签，就删除整个 Session，并从后续合格 Session 递补，最终保留
20,000 个。

完整 Session 按目标事件时间切成 14,000/3,000/3,000 个训练、验证和测试 Session，
再对每个 Session 留出最后一个事件作为标签。针对每个目标，Session 局部历史只保留
目标前事件；Popularity、ItemCF、Item2Vec 和商品统计使用严格早于目标的日级快照，
并用审计断言检查时间因果性。

召回阶段包含 Recent、Popularity、ItemCF 和 Item2Vec。各路先独立返回候选，再去重
合并；同时记录来源命中、来源排名和 RRF 分数。排序阶段构造召回置信度、商品热度与
转化率、Session 统计、行为重复次数和时间交叉特征。真实候选中非目标商品作为 Hard
Negative。Shared-Bottom共享底层表示并设置三个任务塔，使用任务Mask保证一条样本
只监督其真实目标行为。

测试整体 Candidate Recall 为 `0.9310`；按 clicks/carts/orders 的 `0.1/0.3/0.6`
权重计算，Weighted Candidate Recall 为 `0.9802`。RRF和Shared-Bottom的
Weighted Recall@20分别为`0.7770`和`0.9668`，学习排序明显优于当前启发式融合。

## 必须讲清的口径

- Candidate Recall：目标商品是否出现在合并候选集合中，不限制前 20。
- Weighted Candidate Recall：分别计算三种行为的候选命中率，再按 `0.1/0.3/0.6` 加权。
- Weighted Recall@20：分别计算三种行为的 Recall@20，再按相同权重加权。
- 排序只能改变候选顺序，因此同一行为、同一数据集上的 Recall@20 不得超过对应候选命中率。
- 当前 Top3000 由全量事件频次确定，属于 transductive catalog；严格 Point-in-Time
  覆盖动态召回图、Item2Vec、商品统计和排序特征，不包括目录选择。

## 常见追问

**为什么先切 Session 再筛热门商品？**

Session 边界应由原始连续行为决定。若先过滤商品，被删行为可能人为拉大相邻保留行为的
间隔，从而改变 Session 结构。先切分可保留真实时间边界，商品过滤后再检查最短长度。

**如何证明没有目标泄漏？**

对目标时间 `t`，Session 历史满足 `event_ts < t`；全局快照满足
`event_ts < snapshot_ts <= t`。训练、验证和测试的快照审计均要求
`all_snapshots_causal=true`。当前测试包含 9 个快照，全部通过因果检查。

**Item2Vec如何训练？**

把 Session 看成商品 Token 序列，使用 Skip-Gram Negative Sampling。窗口内中心商品和
上下文商品是正样本，负商品按词频的 `0.75` 次方分布采样。当前配置为 32 维、窗口 2、
5 个负样本、8 轮、batch 1024、Adam 学习率 0.01；clicks/carts/orders 的正样本权重为
`1/3/6`。每个快照仅使用该快照之前的事件训练。

**为什么加购和购买 Recall 很高？**

电商会话内加购和购买经常重复此前浏览或操作过的商品，Recent 与重复次数特征很强。
因此高指标主要反映该数据子集和 Leave-Last-Event-Out 任务中的重复行为，不能外推为
对全量新商品或通用购买预测接近完美。

**为什么选择 Shared-Bottom？**

三个目标共享候选、商品和Session特征，但行为分布和重要性不同。Shared-Bottom用共享层
学习共性表示，再用三个任务塔输出点击、加购和购买分数；任务Mask避免把未观测任务当成
负样本。验证集和测试集Weighted Recall@20分别为`0.9698`和`0.9668`。

**如果扩大数据规模怎么办？**

日级全量重建 ItemCF 和 Item2Vec、完整余弦近邻计算会成为瓶颈。工程上应使用增量统计、
滑动窗口、离线周期更新和 ANN 索引；候选与特征按时间分区存储，线上只读取最新可用版本。
