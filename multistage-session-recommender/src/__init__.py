"""多阶段会话推荐系统的核心源码包。

子包职责：

- ``data``：数据加载、清洗、样本构造和时间切分；
- ``recall``：Recent、Popularity、ItemCF、Item2Vec 四路召回；
- ``features``：Point-in-Time 快照与排序特征构造；
- ``ranking``：标签绑定、负采样、分目标排序模型训练与打分；
- ``evaluation``：候选召回率和 Weighted Recall@K 评估。
"""
