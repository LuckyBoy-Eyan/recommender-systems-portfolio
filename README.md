# 推荐系统算法岗面试作品集

本仓库包含两个定位互补、能够端到端复现的推荐系统项目：

| 项目 | 重点能力 | 真实数据实验结果 |
|---|---|---|
| [MiniTIGER](mini-tiger-generative-recommender/) | 自回归 Semantic ID 生成式推荐 | KuaiRec Recall@20 达到 `0.2137` |
| [多阶段会话推荐](multistage-session-recommender/) | 无泄漏多路召回、特征工程、分目标排序 | RetailRocket Weighted Recall@20 达到 `0.9668` |

两个项目均包含：

- 可端到端运行的训练与评估入口；
- 真实公开数据预处理脚本；
- 固定实验配置与保存的评估指标；
- 核心正确性测试；
- 真实数据实验报告与局限性分析；
- 简历 Bullet、三分钟讲稿及常见面试追问。

公开数据文件默认不提交至 Git。仓库会保存数据准备脚本、固定配置、
实验指标、评估协议和结果限制，保证实验能够复现。

## 仓库结构

```text
.
├── mini-tiger-generative-recommender/  # 生成式推荐：Semantic ID + 自回归解码
├── multistage-session-recommender/     # 多阶段推荐：召回 + 特征 + 多目标排序
├── README.md                            # 作品集总览与阅读入口
└── .gitignore                           # 统一排除数据、模型和本地缓存
```

每个子项目内部统一采用 `configs/`、`docs/`、`results/`、`scripts/`、`src/` 和
`tests/`：
源码与配置负责复现，测试负责正确性，Markdown 报告和轻量 JSON 负责展示实验
结论。原始数据、模型权重、候选明细及环境缓存不属于仓库内容。

## 推荐面试讲解顺序

首先讲解多阶段会话推荐项目，展示推荐系统全链路、数据泄漏意识、召回诊断、
特征工程与多目标排序能力。随后讲解 MiniTIGER，展示生成式推荐、Semantic ID、
自回归建模以及约束解码等前沿方向。

## 验证项目

```bash
cd multistage-session-recommender
python -m pytest -q

cd ../mini-tiger-generative-recommender
python -m pytest -q
```

详细阅读入口：

- [多阶段会话推荐 README](multistage-session-recommender/README.md)
- [多阶段会话推荐演进记录](multistage-session-recommender/docs/project-evolution.md)
- [MiniTIGER README](mini-tiger-generative-recommender/README.md)
- [MiniTIGER 演进记录](mini-tiger-generative-recommender/docs/project-evolution.md)
