# MMoE / PLE V3 Windows训练

解压到项目根目录后运行 `run_windows_rankers_v3.ps1`。

- 完整候选池保存在Parquet中，不使用RRF截断。
- 每个epoch从完整负例池按与RRF无关的哈希均匀抽取约160个负例，epoch间轮换。
- 前90%训练、后10%早停，独立validation只做最终评估。
- 累积标签、MMoE 8 Expert、PLE 2共享+每任务2专属、Embedding 16+8、Dropout 0.2。
- 不读取测试集。
